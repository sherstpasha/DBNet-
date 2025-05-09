import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from utils import replace_bn_gn
from torchvision.ops import DeformConv2d


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, in_channels, num_scales):
        super().__init__()
        # Stage-wise attention
        self.weight_conv = nn.Conv2d(
            in_channels * num_scales, num_scales, kernel_size=1
        )
        self.softmax = nn.Softmax(dim=1)
        # Spatial attention
        self.sa_conv1 = nn.Conv2d(in_channels * num_scales, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sa_conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        # объединяем по каналам
        stacked = torch.cat(feats, dim=1)  # [B, C*num_scales, H, W]
        # Spatial Attention Mask
        mask = self.sigmoid(
            self.sa_conv2(self.relu(self.sa_conv1(stacked)))
        )  # [B,1,H,W]
        # применяем маску к каждому масштабу
        feats = [feat * mask for feat in feats]
        # Stage-wise attention на скорректированных признаках
        stacked = torch.cat(feats, dim=1)
        attn = self.weight_conv(stacked)
        attn = self.softmax(attn)  # [B,num_scales,H,W]
        fused = sum(attn[:, i : i + 1] * feats[i] for i in range(len(feats)))
        return fused


class DeformableConvBlock(nn.Module):
    """
    Обёртка для Conv2d → DeformConv2d с генерацией смещений.
    """

    def __init__(self, conv):
        super().__init__()
        kernel_h, kernel_w = conv.kernel_size
        stride_h, stride_w = conv.stride
        # Генератор смещений: 2 канала на каждую точку ядра (dx, dy)
        self.offset_conv = nn.Conv2d(
            conv.in_channels,
            2 * kernel_h * kernel_w,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=conv.padding,
            bias=True,
        )
        # Deformable Convolution
        self.dcn = DeformConv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=conv.padding,
            bias=(conv.bias is not None),
        )

    def forward(self, x):
        # предсказываем смещения с учётом stride
        offset = self.offset_conv(x)
        # deformable conv принимает input и offset
        return self.dcn(x, offset)


def convert_to_dcn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            # заменяем обычный Conv2d на deformable-блок
            setattr(module, name, DeformableConvBlock(child))
        else:
            convert_to_dcn(child)


def build_model():
    # Backbone: pretrained ResNet50
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    convert_to_dcn(backbone.layer2)
    convert_to_dcn(backbone.layer3)
    convert_to_dcn(backbone.layer4)
    return_layers = {"layer1": "p2", "layer2": "p3", "layer3": "p4", "layer4": "p5"}
    extractor = create_feature_extractor(backbone, return_layers)

    # FPN to unify scales
    fpn = FeaturePyramidNetwork(
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        extra_blocks=None,
    )

    # Adaptive Scale Fusion
    asf = AdaptiveScaleFusion(in_channels=256, num_scales=4)

    # Segmentation head: теперь 4 каналов (score, thresh, boundary, k_map)
    head = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 4, kernel_size=1),  # 4 выходных каналов
    )

    class DBNetPlusPlus(nn.Module):
        def __init__(self, extractor, fpn, asf, head):
            super().__init__()
            self.extractor = extractor
            self.fpn = fpn
            self.asf = asf
            self.head = head

        def forward(self, x):
            feats = self.extractor(x)
            p2, p3, p4, p5 = feats["p2"], feats["p3"], feats["p4"], feats["p5"]

            # FPN outputs as dict with keys '0','1','2','3'
            fpn_feats = self.fpn({"0": p2, "1": p3, "2": p4, "3": p5})

            # Upsample all to p2 size
            target_size = p2.shape[2:]
            upsampled = []
            for idx in ["0", "1", "2", "3"]:
                feat = fpn_feats[idx]
                if feat.shape[2:] != target_size:
                    feat = F.interpolate(
                        feat, size=target_size, mode="bilinear", align_corners=False
                    )
                upsampled.append(feat)

            # Fuse scales
            fused = self.asf(upsampled)

            # Head and final upsample
            out = self.head(fused)
            out = F.interpolate(
                out, size=x.shape[2:], mode="bilinear", align_corners=False
            )  # [B,4,H,W]

            # Разбор каналов
            pl, tl, bl, kl_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
            # Логиты -> вероятности
            pp = torch.sigmoid(pl)
            tp = torch.sigmoid(tl)
            # k_map: применяем softplus, чтобы гарантировать >0
            kl = F.softplus(kl_raw)
            # Binary map с адаптивным k
            binary_map = torch.sigmoid(kl * (pp - tp))

            return {
                "score_logits": pl,
                "thresh_logits": tl,
                "boundary_logits": bl,
                "k_map": kl,
                "binary_map": binary_map,
            }

    model = DBNetPlusPlus(extractor, fpn, asf, head)
    # Replace BatchNorm with GroupNorm for small batch stability
    model = replace_bn_gn(model)
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from utils import replace_bn_gn


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, in_channels, num_scales):
        super().__init__()
        self.weight_conv = nn.Conv2d(
            in_channels * num_scales, num_scales, kernel_size=1
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):
        # feats: list of [B, C, H, W]
        stacked = torch.cat(feats, dim=1)  # [B, C*num_scales, H, W]
        attn = self.weight_conv(stacked)  # [B, num_scales, H, W]
        attn = self.softmax(attn)  # normalize weights across scales
        fused = sum(attn[:, i : i + 1] * feats[i] for i in range(len(feats)))
        return fused


def build_model():
    # Backbone: pretrained ResNet50
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
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

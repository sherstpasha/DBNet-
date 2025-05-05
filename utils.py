import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


def sliding_window_inference(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    window_size=768,
    stride=384,
    db_k=50.0,
):
    # Реализация скользящего окна
    w, h = image.size
    pred_full = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for top in range(0, h - window_size + 1, stride):
        for left in range(0, w - window_size + 1, stride):
            crop = image.crop((left, top, left + window_size, top + window_size))
            inp = normalize(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)["out"]
                pl = out[:, 0:1]
                pp = torch.sigmoid(pl * db_k)
            prob = pp[0, 0].cpu().numpy()
            pred_full[top : top + window_size, left : left + window_size] += prob
            count_map[top : top + window_size, left : left + window_size] += 1
    pred_full = pred_full / np.maximum(count_map, 1)
    return pred_full


def replace_bn_gn(module: nn.Module, num_groups=32):
    # Рекурсивная замена BatchNorm2d на GroupNorm
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = min(num_groups, c)
            while g > 1 and c % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(max(g, 1), c))
        else:
            replace_bn_gn(child, num_groups)
    return module


def hard_negative_mining(bce_map, gt, neg_pos_ratio=3):
    pos = gt.eq(1)
    neg = gt.eq(0)
    npos = pos.sum().item()
    nneg = neg_pos_ratio * npos
    negl = bce_map[neg]
    if npos > 0 and negl.numel() > 0:
        topk = min(int(nneg), negl.numel())
        ln = torch.topk(negl, topk)[0].mean()
    else:
        ln = torch.tensor(0.0, device=bce_map.device)
    lp = bce_map[pos].mean() if npos > 0 else torch.tensor(0.0, device=bce_map.device)
    return lp + ln

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import cv2


def compute_iou(box1, box2):
    """IoU двух прямоугольников [x1,y1,x2,y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    A1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    A2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (A1 + A2 - inter + 1e-6)


def preds_to_boxes(prob_map, thresh=0.5, min_area=10):
    """Из probability map в список bbox [x1,y1,x2,y2]."""
    mask = (prob_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append([x, y, x + w, y + h])
    return boxes


def match_detections(preds, gts, iou_thresh=0.5):
    """TP,FP,FN по предсказанным и gt-боксам."""
    matched, tp = set(), 0
    for pb in preds:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def sliding_window_inference(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    window_size: int = 768,
    stride: int = 384,
    db_k: float = 50.0,
):
    """
    Тилед-инференс с усреднением по окнам.
    Возвращает четыре карты float32 размера H×W:
     - prob_map: финальная вероятность (DBNet)
     - score_map: усреднённые score-логиты
     - thresh_map: усреднённые thresh-логиты
     - bnd_map: усреднённые boundary-логиты
    """
    w, h = image.size
    win_w = min(window_size, w)
    win_h = min(window_size, h)

    # координаты левых верхних углов окон
    xs = (
        [0]
        if w <= win_w
        else list(range(0, w - win_w + 1, stride))
        + ([w - win_w] if (w - win_w) % stride != 0 else [])
    )
    ys = (
        [0]
        if h <= win_h
        else list(range(0, h - win_h + 1, stride))
        + ([h - win_h] if (h - win_h) % stride != 0 else [])
    )

    prob_full = np.zeros((h, w), dtype=np.float32)
    score_full = np.zeros((h, w), dtype=np.float32)
    thresh_full = np.zeros((h, w), dtype=np.float32)
    bnd_full = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    with torch.no_grad():
        for top in ys:
            for left in xs:
                crop = image.crop((left, top, left + win_w, top + win_h))
                inp = normalize(crop).unsqueeze(0).to(device)
                out = model(inp)["out"]  # [1, 3, Wh, Ww]
                sc = out[:, 0:1]  # score-логиты
                th = out[:, 1:2]  # thresh-логиты
                bd = out[:, 2:3]  # boundary-логиты
                pr = torch.sigmoid(db_k * (torch.sigmoid(sc) - torch.sigmoid(th)))

                sc_np = sc[0, 0].cpu().numpy()
                th_np = th[0, 0].cpu().numpy()
                bd_np = bd[0, 0].cpu().numpy()
                pr_np = pr[0, 0].cpu().numpy()

                score_full[top : top + win_h, left : left + win_w] += sc_np
                thresh_full[top : top + win_h, left : left + win_w] += th_np
                bnd_full[top : top + win_h, left : left + win_w] += bd_np
                prob_full[top : top + win_h, left : left + win_w] += pr_np
                count_map[top : top + win_h, left : left + win_w] += 1

    denom = np.maximum(count_map, 1.0)
    return prob_full / denom, score_full / denom, thresh_full / denom, bnd_full / denom


def replace_bn_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    Recursively replace all BatchNorm2d layers in `module` with GroupNorm.
    Uses up to `num_groups` groups (or fewer if num_features % num_groups != 0).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = min(num_groups, c)
            # ensure g divides c
            while g > 1 and c % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(num_groups=g, num_channels=c))
        else:
            replace_bn_gn(child, num_groups)
    return module


def hard_negative_mining(
    bce_map: torch.Tensor, gt: torch.Tensor, neg_pos_ratio: int = 3
) -> torch.Tensor:
    """
    Given a per-pixel BCE loss map and a binary ground-truth mask (1=text, 0=background),
    computes the average loss over all positives plus the top-k hardest negatives,
    where k = neg_pos_ratio * (#positives). If no positives, uses all negatives.
    """
    pos_mask = gt.eq(1)
    neg_mask = gt.eq(0)
    n_pos = pos_mask.sum().item()
    n_neg_total = neg_mask.sum().item()
    # positive loss
    pos_loss = (
        bce_map[pos_mask].mean()
        if n_pos > 0
        else torch.tensor(0.0, device=bce_map.device)
    )

    # negative loss: top-k hardest
    if n_pos > 0 and n_neg_total > 0:
        k = min(int(neg_pos_ratio * n_pos), n_neg_total)
        neg_vals = bce_map[neg_mask]
        neg_loss = torch.topk(neg_vals, k)[0].mean()
    elif n_neg_total > 0:
        # fallback: use all negatives
        neg_loss = bce_map[neg_mask].mean()
    else:
        neg_loss = torch.tensor(0.0, device=bce_map.device)

    return pos_loss + neg_loss

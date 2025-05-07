import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


def sliding_window_inference(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    window_size: int = 768,
    stride: int = 384,
    db_k: float = 50.0,
):
    """
    Tiled inference over `image` с overlap, возвращает четыре карты float32 H×W:
      - prob_full:  финальная probability map (sigmoid(k*(score-thresh)))
      - score_full: среднее по плиткам значение score-логитов
      - thresh_full: среднее по плиткам значение thresh-логитов
      - bnd_full:   среднее по плиткам значение boundary-логитов
    """
    w, h = image.size
    win_w = min(window_size, w)
    win_h = min(window_size, h)

    # позиции тайлов
    xs = list(range(0, w - win_w + 1, stride)) if w > win_w else [0]
    if xs[-1] != w - win_w: xs.append(w - win_w)
    ys = list(range(0, h - win_h + 1, stride)) if h > win_h else [0]
    if ys[-1] != h - win_h: ys.append(h - win_h)

    pred_full   = np.zeros((h, w), dtype=np.float32)
    score_full  = np.zeros((h, w), dtype=np.float32)
    thresh_full = np.zeros((h, w), dtype=np.float32)
    bnd_full    = np.zeros((h, w), dtype=np.float32)
    count_map   = np.zeros((h, w), dtype=np.float32)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    model.eval()
    with torch.no_grad():
        for top in ys:
            for left in xs:
                crop = image.crop((left, top, left + win_w, top + win_h))
                inp  = normalize(crop).unsqueeze(0).to(device)

                out = model(inp)["out"]
                # out: [B,3,Wh,Ww] → три логита
                score = out[:,0:1]   # score-логиты
                thresh= out[:,1:2]   # thresh-логиты
                bnd   = out[:,2:3]   # boundary-логиты

                prob  = torch.sigmoid(db_k * (score - thresh))

                sf = score[0,0].cpu().numpy()
                tf = thresh[0,0].cpu().numpy()
                bf = bnd[0,0].cpu().numpy()
                pf = prob[0,0].cpu().numpy()

                score_full [top:top+win_h, left:left+win_w] += sf
                thresh_full[top:top+win_h, left:left+win_w] += tf
                bnd_full   [top:top+win_h, left:left+win_w] += bf
                pred_full  [top:top+win_h, left:left+win_w] += pf
                count_map  [top:top+win_h, left:left+win_w] += 1

    # Нормировка по числу влившихся тайлов
    count_map = np.maximum(count_map, 1.0)
    return (
        pred_full   / count_map,
        score_full  / count_map,
        thresh_full / count_map,
        bnd_full    / count_map,
    )


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
    bce_map: torch.Tensor,
    gt: torch.Tensor,
    neg_pos_ratio: int = 3
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
    pos_loss = bce_map[pos_mask].mean() if n_pos > 0 else torch.tensor(0.0, device=bce_map.device)

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

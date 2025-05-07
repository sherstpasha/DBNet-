import os
import random
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR

from dataset import COCOTextDataset
from model import build_model
from utils import hard_negative_mining, sliding_window_inference


def compute_iou(box1, box2):
    """IoU двух прямоугольников [x1,y1,x2,y2]."""
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    A1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    A2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (A1 + A2 - inter + 1e-6)


def preds_to_boxes(prob_map, thresh=0.5, min_area=10):
    """Из probability map в список bbox [x1,y1,x2,y2]."""
    mask = (prob_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append([x,y,x+w,y+h])
    return boxes


def match_detections(preds, gts, iou_thresh=0.5):
    """TP,FP,FN по предсказанным и gt-боксам."""
    matched, tp = set(), 0
    for pb in preds:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in matched: continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gts)   - tp
    return tp, fp, fn


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min–max нормализация в [0,255] uint8."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).astype(np.uint8)


def train(args):
    # ----------------------
    # 1) Train/Val datasets
    # ----------------------
    train_ds = COCOTextDataset(
        images_dirs=args.train_dirs,
        ann_files=args.train_anns,
        base_size=args.base_size,
        crop_range=tuple(args.crop_range),
        p_flip=args.p_flip,
        p_rotate=args.p_rotate,
        p_color=args.p_color,
        max_angle=args.max_angle,
        empty_thresh=args.empty_thresh,
    )
    if args.val_dirs and args.val_anns:
        val_ds = COCOTextDataset(
            images_dirs=args.val_dirs,
            ann_files=args.val_anns,
            base_size=args.base_size,
            crop_range=None,  # full-image
            p_flip=0.0, p_rotate=0.0, p_color=0.0,
            max_angle=0.0, empty_thresh=0.0,
        )
    else:
        n_val = int(len(train_ds) * args.val_split)
        train_ds, val_ds = random_split(train_ds, [len(train_ds)-n_val, n_val])

    tr_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # -------------------------------
    # 2) Model, optimizer, scheduler
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(db_k=args.db_k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=len(tr_loader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=args.max_lr/args.lr,
        final_div_factor=1e4,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    l1  = nn.L1Loss(reduction="none")

    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_loss = float("inf")
    global_step = 0

    # ---------------
    # 3) Training loop
    # ---------------
    for ep in range(1, args.epochs+1):
        # — TRAIN —
        model.train()
        train_loss = 0.0
        for imgs, sgt, tgt, bgt, _, _ in tr_loader:
            imgs, sgt, tgt, bgt = [t.to(device) for t in (imgs, sgt, tgt, bgt)]
            optimizer.zero_grad()

            out = model(imgs)["out"]
            pl, tl, bl = out[:,0:1], out[:,1:2], out[:,2:3]
            pp, tp_   = torch.sigmoid(pl), torch.sigmoid(tl)
            dbp       = torch.sigmoid(model.db_k * (pp - tp_))

            Ls   = hard_negative_mining(bce(pl, sgt), sgt)
            Lb   = bce(dbp, sgt).mean()
            Lt   = (l1(tp_, tgt)*(tgt>0)).sum()/((tgt>0).sum()+1e-6)
            Lbnd = bce(bl, bgt).mean()
            loss = Ls + args.alpha*Lb + args.beta*Lt + args.gamma*Lbnd

            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            train_loss += loss.item()
            global_step += 1

        writer.add_scalar("Loss/train_epoch", train_loss/len(tr_loader), ep)

        # — VALIDATION каждые N эпох —
        if ep % args.val_every == 0:
            model.eval()
            val_loss = 0.0
            tp_sum = fp_sum = fn_sum = 0
            saved_val = []

            with torch.no_grad():
                for idx in range(len(val_ds)):
                    img_t, sgt, tgt, bgt, raw, gt_boxes = val_ds[idx]
                    raw_np = raw.permute(1,2,0).cpu().numpy()

                    # tiled inference сразу четыре карты
                    prob_map, score_map, thresh_map, bnd_map = sliding_window_inference(
                        Image.fromarray(raw_np), model, device,
                        window_size=args.window_size,
                        stride=args.stride,
                        db_k=args.db_k
                    )

                    # расчёт сегментационного лосса на CPU
                    sc = torch.from_numpy(score_map).unsqueeze(0)
                    th = torch.from_numpy(thresh_map).unsqueeze(0)
                    bd = torch.from_numpy(bnd_map).unsqueeze(0)
                    Ls   = hard_negative_mining(bce(sc.unsqueeze(0), sgt.unsqueeze(0)), sgt.unsqueeze(0))
                    Lb   = bce(torch.sigmoid(args.db_k*(torch.sigmoid(sc)-torch.sigmoid(th))).unsqueeze(0),
                               sgt.unsqueeze(0)).mean()
                    Lt   = (l1(torch.sigmoid(th).unsqueeze(0), tgt.unsqueeze(0)) * (tgt>0).unsqueeze(0)).sum() \
                           / ((tgt>0).sum()+1e-6)
                    Lbnd = bce(bd.unsqueeze(0), bgt.unsqueeze(0)).mean()
                    val_loss += (Ls + args.alpha*Lb + args.beta*Lt + args.gamma*Lbnd).item()

                    # detection
                    preds = preds_to_boxes(prob_map, thresh=args.det_thresh, min_area=args.min_area)
                    gts   = gt_boxes.cpu().tolist()
                    tp,fp,fn = match_detections(preds, gts, iou_thresh=args.iou_thresh)
                    tp_sum += tp; fp_sum += fp; fn_sum += fn

                    if len(saved_val) < args.num_vis:
                        saved_val.append((raw_np, gts, preds, prob_map))

            avg_val_loss = val_loss / len(val_ds)
            writer.add_scalar("Loss/val_epoch", avg_val_loss, ep)

            # сохранить лучшую модель
            os.makedirs(args.ckpt_dir, exist_ok=True)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))

            # метрики детекции
            precision = tp_sum/(tp_sum+fp_sum+1e-6)
            recall    = tp_sum/(tp_sum+fn_sum+1e-6)
            f1        = 2*precision*recall/(precision+recall+1e-6)
            writer.add_scalar("Metrics/Precision", precision, ep)
            writer.add_scalar("Metrics/Recall",    recall,    ep)
            writer.add_scalar("Metrics/F1",        f1,        ep)

            # визуализация
            for i, (raw_np, gts, preds, prob_map) in enumerate(saved_val):
                vis = raw_np.copy()
                for x1,y1,x2,y2 in gts:
                    cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                for x1,y1,x2,y2 in preds:
                    cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                writer.add_image(
                    f"Val/Boxes/{i}",
                    torch.from_numpy(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
                         .permute(2,0,1),
                    ep, dataformats="CHW"
                )
                writer.add_image(
                    f"Val/Prob_Map/{i}",
                    torch.from_numpy(normalize_to_uint8(prob_map)).unsqueeze(0),
                    ep, dataformats="CHW"
                )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs",   nargs="+", required=True)
    parser.add_argument("--train_anns",   nargs="+", required=True)
    parser.add_argument("--val_dirs",     nargs="+")
    parser.add_argument("--val_anns",     nargs="+")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--max_lr",       type=float, default=1e-3)
    parser.add_argument("--val_split",    type=float, default=0.1)
    parser.add_argument("--base_size",    type=int,   default=2048)
    parser.add_argument("--crop_range",   nargs=2,    type=int,   default=(512,512))
    parser.add_argument("--p_flip",       type=float, default=0.1)
    parser.add_argument("--p_rotate",     type=float, default=0.1)
    parser.add_argument("--p_color",      type=float, default=0.1)
    parser.add_argument("--max_angle",    type=float, default=7.0)
    parser.add_argument("--empty_thresh", type=float, default=0.01)
    parser.add_argument("--db_k",         type=float, default=50.0)
    parser.add_argument("--alpha",        type=float, default=1.0)
    parser.add_argument("--beta",         type=float, default=5.0)
    parser.add_argument("--gamma",        type=float, default=1.0)
    parser.add_argument("--det_thresh",   type=float, default=0.5)
    parser.add_argument("--iou_thresh",   type=float, default=0.5)
    parser.add_argument("--min_area",     type=int,   default=10)
    parser.add_argument("--window_size",  type=int,   default=512)
    parser.add_argument("--stride",       type=int,   default=256)
    parser.add_argument("--num_vis",      type=int,   default=3)
    parser.add_argument("--val_every",    type=int,   default=10,
                        help="проводить валидацию каждые N эпох")
    parser.add_argument("--log_dir",      type=str,   default="runs")
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints")
    args = parser.parse_args()

    train(args)
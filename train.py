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
    """Compute IoU between two boxes: [x1,y1,x2,y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interA = interW * interH
    A1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    A2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return interA / (A1 + A2 - interA + 1e-6)


def preds_to_boxes(prob_map, thresh=0.5, min_area=10):
    """Convert probability map (H,W) to list of boxes [x1,y1,x2,y2]."""
    mask = (prob_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            boxes.append([x, y, x + w, y + h])
    return boxes


def match_detections(preds, gts, iou_thresh=0.5):
    """Match predicted boxes to GT boxes, return TP, FP, FN counts."""
    matched, tp = set(), 0
    for pb in preds:
        best_i, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_i:
                best_i, best_j = iou, j
        if best_i >= iou_thresh and best_j >= 0:
            tp += 1
            matched.add(best_j)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min–max normalize to [0,255] uint8."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255).astype(np.uint8)


def train(args):
    # --- 1) Датасеты ---
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
            p_flip=0.0,
            p_rotate=0.0,
            p_color=0.0,
            max_angle=0.0,
            empty_thresh=0.0,
        )
    else:
        n_val = int(len(train_ds) * args.val_split)
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])

    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- 2) Модель, оптимизатор, LR-сcheduler ---
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
        div_factor=args.max_lr / args.lr,
        final_div_factor=1e4,
    )

    bce = nn.BCEWithLogitsLoss(reduction="none")
    l1 = nn.L1Loss(reduction="none")
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    for ep in range(1, args.epochs + 1):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        for imgs, sgt, tgt, bgt, raw, _ in tr_loader:
            imgs, sgt, tgt, bgt = [t.to(device) for t in (imgs, sgt, tgt, bgt)]
            optimizer.zero_grad()
            out = model(imgs)["out"]
            pl, tl, bl = out[:, 0:1], out[:, 1:2], out[:, 2:3]
            pp, tp_ = torch.sigmoid(pl), torch.sigmoid(tl)
            dbp = torch.sigmoid(model.db_k * (pp - tp_))

            Ls = hard_negative_mining(bce(pl, sgt), sgt)
            Lb = bce(dbp, sgt).mean()
            Lt = (l1(tp_, tgt) * (tgt > 0)).sum() / ((tgt > 0).sum() + 1e-6)
            Lbnd = bce(bl, bgt).mean()
            loss = Ls + args.alpha * Lb + args.beta * Lt + args.gamma * Lbnd

            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            running_loss += loss.item()
            global_step += 1

        writer.add_scalar("Loss/train_epoch", running_loss / len(tr_loader), ep)

        # --- VISUALIZE ON TRAIN SUBSET ---
        model.eval()
        vis_idxs = random.sample(range(len(train_ds)), min(args.num_vis, len(train_ds)))
        train_vis = []
        with torch.no_grad():
            for idx in vis_idxs:
                img_t, score_t, thresh_t, bnd_t, raw, gt_boxes = train_ds[idx]
                raw_np = raw.permute(1, 2, 0).cpu().numpy()
                inp = img_t.unsqueeze(0).to(device)
                out = model(inp)["out"]
                pl, tl, _ = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                pp, tp_ = torch.sigmoid(pl), torch.sigmoid(tl)
                dbp_map = dbp = (
                    torch.sigmoid(model.db_k * (pp - tp_))[0, 0].cpu().numpy()
                )

                # «сырые» логиты
                pl_map = pl[0, 0].cpu().numpy()
                tl_map = tl[0, 0].cpu().numpy()

                preds = preds_to_boxes(
                    dbp_map, thresh=args.det_thresh, min_area=args.min_area
                )
                gt_mask = (score_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pred_mask = (dbp_map > args.det_thresh).astype(np.uint8) * 255

                train_vis.append(
                    (
                        raw_np,
                        gt_boxes.cpu().tolist(),
                        preds,
                        gt_mask,
                        pred_mask,
                        normalize_to_uint8(pl_map),
                        normalize_to_uint8(tl_map),
                        normalize_to_uint8(dbp_map),
                    )
                )

        for i, (raw_np, gts, preds, gt_m, pred_m, pl_u8, tl_u8, db_u8) in enumerate(
            train_vis
        ):
            # A) Raw + all boxes
            vis = raw_np.copy()
            for x1, y1, x2, y2 in gts:
                cv2.rectangle(
                    vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )
            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            writer.add_image(
                f"Train/Boxes/{i}",
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                ep,
                dataformats="HWC",
            )

            # B) GT mask
            writer.add_image(
                f"Train/GT_Mask/{i}", np.stack([gt_m] * 3).transpose(2, 0, 1), ep
            )

            # C) Pred mask
            writer.add_image(
                f"Train/Pred_Mask/{i}", np.stack([pred_m] * 3).transpose(2, 0, 1), ep
            )

            # D) Only Pred boxes
            vis2 = raw_np.copy()
            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    vis2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            writer.add_image(
                f"Train/Pred_Boxes/{i}",
                cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB),
                ep,
                dataformats="HWC",
            )

            # E) Raw score-logit heatmap
            writer.add_image(f"Train/Raw_Score/{i}", pl_u8[np.newaxis, :, :], ep)

            # F) Raw thresh-logit heatmap
            writer.add_image(f"Train/Raw_Thresh/{i}", tl_u8[np.newaxis, :, :], ep)

            # G) Final probability heatmap
            writer.add_image(f"Train/Heatmap/{i}", db_u8[np.newaxis, :, :], ep)

        # --- VALIDATION VIA TILED INFERENCE ---
        model.eval()
        tp_sum = fp_sum = fn_sum = 0
        val_vis = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                img_t, score_t, thresh_t, bnd_t, raw_img, gt_boxes = val_ds[idx]
                raw_np = raw_img.permute(1, 2, 0).cpu().numpy()
                pil = Image.fromarray(raw_np)

                prob_map = sliding_window_inference(
                    pil,
                    model,
                    device,
                    window_size=args.window_size,
                    stride=args.stride,
                    db_k=args.db_k,
                )
                preds = preds_to_boxes(
                    prob_map, thresh=args.det_thresh, min_area=args.min_area
                )
                gt_m = (score_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pred_m = (prob_map > args.det_thresh).astype(np.uint8) * 255

                # для «сырых» карт нужно сделать отдельный проход:
                inp = img_t.unsqueeze(0).to(device)
                out = model(inp)["out"]
                pl, tl, _ = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                pp, tp_ = torch.sigmoid(pl), torch.sigmoid(tl)
                db_raw = torch.sigmoid(model.db_k * (pp - tp_))[0, 0].cpu().numpy()
                pl_map = pl[0, 0].cpu().numpy()
                tl_map = tl[0, 0].cpu().numpy()

                tp, fp, fn = match_detections(
                    preds, gt_boxes.cpu().tolist(), args.iou_thresh
                )
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

                if len(val_vis) < args.num_vis:
                    val_vis.append(
                        (
                            raw_np,
                            gt_boxes.cpu().tolist(),
                            preds,
                            gt_m,
                            pred_m,
                            normalize_to_uint8(pl_map),
                            normalize_to_uint8(tl_map),
                            normalize_to_uint8(db_raw),
                        )
                    )

        # метрики
        prec = tp_sum / (tp_sum + fp_sum + 1e-6)
        rec = tp_sum / (tp_sum + fn_sum + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        writer.add_scalar("Metrics/Precision", prec, ep)
        writer.add_scalar("Metrics/Recall", rec, ep)
        writer.add_scalar("Metrics/F1", f1, ep)

        # визуализация VAL
        for i, (raw_np, gts, preds, gt_m, pred_m, pl_u8, tl_u8, db_u8) in enumerate(
            val_vis
        ):
            # A) Raw + boxes
            vis = raw_np.copy()
            for x1, y1, x2, y2 in gts:
                cv2.rectangle(
                    vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )
            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            writer.add_image(
                f"Val/Boxes/{i}",
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                ep,
                dataformats="HWC",
            )

            # B) GT mask
            writer.add_image(
                f"Val/GT_Mask/{i}", np.stack([gt_m] * 3).transpose(2, 0, 1), ep
            )

            # C) Pred mask
            writer.add_image(
                f"Val/Pred_Mask/{i}", np.stack([pred_m] * 3).transpose(2, 0, 1), ep
            )

            # D) Only Pred boxes
            vis2 = raw_np.copy()
            for x1, y1, x2, y2 in preds:
                cv2.rectangle(
                    vis2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            writer.add_image(
                f"Val/Pred_Boxes/{i}",
                cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB),
                ep,
                dataformats="HWC",
            )

            # E) Raw score-logit
            writer.add_image(f"Val/Raw_Score/{i}", pl_u8[np.newaxis, :, :], ep)

            # F) Raw thresh-logit
            writer.add_image(f"Val/Raw_Thresh/{i}", tl_u8[np.newaxis, :, :], ep)

            # G) Final heatmap
            writer.add_image(f"Val/Heatmap/{i}", db_u8[np.newaxis, :, :], ep)

        # --- Сохранение чекпойнта ---
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"ckpt_ep{ep}.pth"))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--train_anns", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+")
    parser.add_argument("--val_anns", nargs="+")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--base_size", type=int, default=2048)
    parser.add_argument("--crop_range", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--p_flip", type=float, default=0.1)
    parser.add_argument("--p_rotate", type=float, default=0.1)
    parser.add_argument("--p_color", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=7.0)
    parser.add_argument("--empty_thresh", type=float, default=0.01)
    parser.add_argument("--db_k", type=float, default=50.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num_vis", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

from dataset import COCOTextDataset
from model import build_model
from utils import (
    hard_negative_mining,
    sliding_window_inference,
)  # sliding_window_inference доступен для большого изображения :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}


def train(args):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    # 1) TRAIN dataset (c кропами)
    train_ds = COCOTextDataset(
        images_dirs=args.train_dirs,
        ann_files=args.train_anns,
        base_size=args.base_size,
        crop_size=args.crop_size,
        shrink_ratio=args.shrink_ratio,
        p_flip=args.p_flip,
        p_rotate=args.p_rotate,
        p_color=args.p_color,
        max_angle=args.max_angle,
    )
    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2) VAL dataset (полные кадры)
    val_loader = None
    if args.val_dirs and args.val_anns:
        val_ds = COCOTextDataset(
            images_dirs=args.val_dirs,
            ann_files=args.val_anns,
            base_size=args.base_size,
            crop_size=None,
            shrink_ratio=args.shrink_ratio,
            p_flip=0.0,
            p_rotate=0.0,
            p_color=0.0,
            max_angle=0.0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # 3) MODEL / OPT / SCHEDULER
    model = build_model().to(device)
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
    scaler = GradScaler()
    global_step = 0

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for ep in range(1, args.epochs + 1):
        # ---- TRAIN ----
        model.train()
        sum_Ls = sum_Lb = sum_Lt = sum_Lbnd = sum_loss = 0.0

        # placeholders for last batch
        last_imgs = last_score_gt = last_thresh_gt = last_bnd_gt = None
        last_pl = last_tl = last_bl = last_kl = None

        for img, score_gt, thresh_gt, bnd_gt in tr_loader:
            img = img.to(device, non_blocking=True)
            score_gt = score_gt.to(device, non_blocking=True)
            thresh_gt = thresh_gt.to(device, non_blocking=True)
            bnd_gt = bnd_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                out = model(img)
                pl, tl, bl, kl, bp = (
                    out["score_logits"],
                    out["thresh_logits"],
                    out["boundary_logits"],
                    out["k_map"],
                    out["binary_map"],
                )
                Ls = hard_negative_mining(bce(pl, score_gt), score_gt)
                Lb = bce(bp, score_gt).mean()
                Lt = (l1(torch.sigmoid(tl), thresh_gt) * (thresh_gt > 0)).sum() / (
                    (thresh_gt > 0).sum() + 1e-6
                )
                Lbnd = bce(bl, bnd_gt).mean()
                loss = Ls + args.alpha * Lb + args.beta * Lt + args.gamma * Lbnd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler._get_lr_called_within_step = True
            scheduler.step()

            # logging
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            writer.add_scalar("Loss/train_score", Ls.item(), global_step)
            writer.add_scalar("Loss/train_binary", Lb.item(), global_step)
            writer.add_scalar("Loss/train_thresh", Lt.item(), global_step)
            writer.add_scalar("Loss/train_boundary", Lbnd.item(), global_step)
            writer.add_scalar("k_map/mean", kl.mean().item(), global_step)
            writer.add_scalar("k_map/std", kl.std().item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            sum_Ls += Ls.item()
            sum_Lb += Lb.item()
            sum_Lt += Lt.item()
            sum_Lbnd += Lbnd.item()
            sum_loss += loss.item()
            global_step += 1

            # сохранить для визуализации (detach + cpu)
            last_imgs = img.detach().cpu()
            last_score_gt = score_gt[0, 0].detach().cpu()
            last_thresh_gt = thresh_gt[0, 0].detach().cpu()
            last_bnd_gt = bnd_gt[0, 0].detach().cpu()
            # приводим логи­­ты и k_map к float32, чтобы sigmoid на CPU работал
            last_pl = pl[0].detach().cpu().float()
            last_tl = tl[0].detach().cpu().float()
            last_bl = bl[0].detach().cpu().float()
            last_kl = kl[0].detach().cpu().float()

        # epoch-wise train metrics
        nb = len(tr_loader)
        writer.add_scalar("Loss/train_total_epoch", sum_loss / nb, ep)
        writer.add_scalar("Loss/train_score_epoch", sum_Ls / nb, ep)
        writer.add_scalar("Loss/train_binary_epoch", sum_Lb / nb, ep)
        writer.add_scalar("Loss/train_thresh_epoch", sum_Lt / nb, ep)
        writer.add_scalar("Loss/train_boundary_epoch", sum_Lbnd / nb, ep)

        # очистка памяти после train
        del img, score_gt, thresh_gt, bnd_gt, pl, tl, bl, kl, bp, loss
        torch.cuda.empty_cache()

        # визуализация последнего train-батча
        model.eval()
        with torch.no_grad():
            pr = torch.sigmoid(
                last_kl * (torch.sigmoid(last_pl) - torch.sigmoid(last_tl))
            )
            writer.add_image("Train/Input_last", last_imgs[0], ep)
            writer.add_image("Train/ScoreProb_last", torch.sigmoid(last_pl), ep)
            writer.add_image("Train/ThreshProb_last", torch.sigmoid(last_tl), ep)
            writer.add_image("Train/BoundaryProb_last", torch.sigmoid(last_bl), ep)
            writer.add_image("Train/k_map_last", last_kl, ep)
            writer.add_image("Train/Pred_last", pr, ep)
            writer.add_image("Train/GT_score_last", last_score_gt.unsqueeze(0), ep)
            writer.add_image("Train/GT_thresh_last", last_thresh_gt.unsqueeze(0), ep)
            writer.add_image("Train/GT_bnd_last", last_bnd_gt.unsqueeze(0), ep)

        # ---- VALIDATION ----
        if val_loader is not None:
            sum_vLs = sum_vLb = sum_vLt = sum_vLbnd = sum_vloss = 0.0
            val_img_raw = None

            model.eval()
            with torch.no_grad():
                # 1) пробег по кропам для метрик
                for i, (img_v, score_v, thresh_v, bnd_v) in enumerate(val_loader):
                    img_v = img_v.to(device, non_blocking=True)
                    score_v = score_v.to(device, non_blocking=True)
                    thresh_v = thresh_v.to(device, non_blocking=True)
                    bnd_v = bnd_v.to(device, non_blocking=True)

                    # forward на кропе (AMP не нужен в валидации, но можно обернуть)
                    out_v = model(img_v)
                    pl_v, tl_v, bl_v, kl_v, bp_v = (
                        out_v["score_logits"],
                        out_v["thresh_logits"],
                        out_v["boundary_logits"],
                        out_v["k_map"],
                        out_v["binary_map"],
                    )

                    # считаем лоссы
                    vLs = hard_negative_mining(bce(pl_v, score_v), score_v)
                    vLb = bce(bp_v, score_v).mean()
                    vLt = (l1(torch.sigmoid(tl_v), thresh_v) * (thresh_v > 0)).sum() / (
                        (thresh_v > 0).sum() + 1e-6
                    )
                    vLbnd = bce(bl_v, bnd_v).mean()
                    vloss = (
                        vLs + args.alpha * vLb + args.beta * vLt + args.gamma * vLbnd
                    )

                    sum_vLs += vLs.item()
                    sum_vLb += vLb.item()
                    sum_vLt += vLt.item()
                    sum_vLbnd += vLbnd.item()
                    sum_vloss += vloss.item()

                    if i == 0:
                        # сохраним исходное полное PIL-изображение для слайдов
                        img_cpu = img_v[0].detach().cpu()
                        inv = inv_normalize(img_cpu)
                        val_img_raw = transforms.ToPILImage()(inv)

                # усреднённый лосс
                nvb = len(val_loader)
                avg_val_loss = sum_vloss / nvb
                writer.add_scalar("Loss/val_total_epoch", avg_val_loss, ep)
                writer.add_scalar("Loss/val_score_epoch", sum_vLs / nvb, ep)
                writer.add_scalar("Loss/val_binary_epoch", sum_vLb / nvb, ep)
                writer.add_scalar("Loss/val_thresh_epoch", sum_vLt / nvb, ep)
                writer.add_scalar("Loss/val_boundary_epoch", sum_vLbnd / nvb, ep)

                # чекпоинт по лучшей метрике
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.ckpt_dir, "best_model.pth"),
                    )
                    print(f"[Epoch {ep}] New best val loss: {avg_val_loss:.4f}")

                # 2) sliding-window inference на полную карту
                prob_full, sc_full, th_full, bd_full = sliding_window_inference(
                    val_img_raw,
                    model,
                    device,
                    window_size=args.window_size,
                    stride=args.stride,
                )

                # логируем результаты слайдов
                writer.add_image(
                    "Val/Input_full", transforms.ToTensor()(val_img_raw), ep
                )
                writer.add_image(
                    "Val/Prob_full", torch.from_numpy(prob_full).unsqueeze(0), ep
                )
                writer.add_image(
                    "Val/Score_full", torch.from_numpy(sc_full).unsqueeze(0), ep
                )
                writer.add_image(
                    "Val/Thresh_full", torch.from_numpy(th_full).unsqueeze(0), ep
                )
                writer.add_image(
                    "Val/Boundary_full", torch.from_numpy(bd_full).unsqueeze(0), ep
                )

            # очистка GPU-памяти
            del img_v, score_v, thresh_v, bnd_v, pl_v, tl_v, bl_v, kl_v, bp_v, vloss
            torch.cuda.empty_cache()
            model.train()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--train_anns", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", default=None)
    parser.add_argument("--val_anns", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--base_size", type=int, default=2048)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--shrink_ratio", type=float, default=0.3)
    parser.add_argument("--p_flip", type=float, default=0.1)
    parser.add_argument("--p_rotate", type=float, default=0.1)
    parser.add_argument("--p_color", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=7.0)
    parser.add_argument("--window_size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args)

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# Обратная нормализация для вывода изображений
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

from dataset import COCOTextDataset
from model import build_model
from utils import hard_negative_mining, sliding_window_inference


def train(args):
    # Настройка устройства
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Создание папки для чекпойнтов
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    # ========== DATASET / DATALOADER ==========
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
            pin_memory=False,
        )

    # ========== MODEL / OPT / SCHEDULER ==========
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = GradScaler()

    # ========== TENSORBOARD ==========
    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        sum_Ls = sum_Lb = sum_Lt = sum_Lbnd = sum_loss = 0.0

        # --- Training loop ---
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
                Ls = hard_negative_mining(
                    nn.functional.binary_cross_entropy_with_logits(
                        pl, score_gt, reduction="none"
                    ),
                    score_gt,
                )
                Lb = nn.functional.binary_cross_entropy_with_logits(bp, score_gt).mean()
                Lt = (
                    nn.functional.l1_loss(
                        torch.sigmoid(tl), thresh_gt, reduction="none"
                    )
                    * (thresh_gt > 0)
                ).sum() / ((thresh_gt > 0).sum() + 1e-6)
                Lbnd = nn.functional.binary_cross_entropy_with_logits(bl, bnd_gt).mean()
                loss = Ls + args.alpha * Lb + args.beta * Lt + args.gamma * Lbnd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Логгируем метрики
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            writer.add_scalar("Loss/train_score", Ls.item(), global_step)
            writer.add_scalar("Loss/train_binary", Lb.item(), global_step)
            writer.add_scalar("Loss/train_thresh", Lt.item(), global_step)
            writer.add_scalar("Loss/train_boundary", Lbnd.item(), global_step)
            writer.add_scalar("k_map/mean", kl.mean().item(), global_step)
            writer.add_scalar("k_map/std", kl.std().item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar(
                "Data/train_pos_pixel_ratio", score_gt.mean().item(), global_step
            )

            sum_Ls += Ls.item()
            sum_Lb += Lb.item()
            sum_Lt += Lt.item()
            sum_Lbnd += Lbnd.item()
            sum_loss += loss.item()
            global_step += 1

            last_imgs = img.cpu()
            last_score_gt = score_gt[0, 0].cpu()
            last_thresh_gt = thresh_gt[0, 0].cpu()
            last_bnd_gt = bnd_gt[0, 0].cpu()
            last_pl = pl[0].float().cpu()
            last_tl = tl[0].float().cpu()
            last_bl = bl[0].float().cpu()
            last_kl = kl[0].float().cpu()

        # Эпоховые метрики
        nb = len(tr_loader)
        writer.add_scalar("Loss/train_total_epoch", sum_loss / nb, ep)
        writer.add_scalar("Loss/train_score_epoch", sum_Ls / nb, ep)
        writer.add_scalar("Loss/train_binary_epoch", sum_Lb / nb, ep)
        writer.add_scalar("Loss/train_thresh_epoch", sum_Lt / nb, ep)
        writer.add_scalar("Loss/train_boundary_epoch", sum_Lbnd / nb, ep)

        # Логируем гистограммы весов и градиентов раз в эпоху
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param.detach().cpu().numpy(), ep)
            if param.grad is not None:
                writer.add_histogram(
                    f"Grads/{name}", param.grad.detach().cpu().numpy(), ep
                )

        # --- Отрисовка последнего кропа ---
        model.eval()
        with torch.no_grad():
            pr = torch.sigmoid(
                last_kl * (torch.sigmoid(last_pl) - torch.sigmoid(last_tl))
            )
            gt_rgb = torch.stack([last_score_gt, last_thresh_gt, last_bnd_gt], dim=0)
            writer.add_image("Train/Input_last", last_imgs[0], ep)
            writer.add_image("Train/ScoreLogits_last", last_pl, ep)
            writer.add_image("Train/ThreshLogits_last", last_tl, ep)
            writer.add_image("Train/BoundaryLogits_last", last_bl, ep)
            writer.add_image("Train/k_map_last", last_kl, ep)
            writer.add_image("Train/Pred_last", pr, ep)
            writer.add_image("Train/GT_last", gt_rgb, ep)

        # --- VALIDATION ---
        if val_loader is not None:
            sum_vLs = sum_vLb = sum_vLt = sum_vLbnd = sum_vloss = 0.0
            sum_thresh_mse = sum_thresh_mae = 0.0
            all_gt_labels, all_pred_probs = [], []
            all_gt_labels_bnd, all_pred_probs_bnd = [], []
            all_gt_labels_bin, all_pred_probs_bin = [], []
            all_k_preds = []
            pos_ratio_sum = 0.0
            val_img_raw = None

            with torch.no_grad():
                for i, (img_v, score_v, thresh_v, bnd_v) in enumerate(val_loader):
                    img_v = img_v.to(device, non_blocking=True)
                    score_v = score_v.to(device, non_blocking=True)
                    thresh_v = thresh_v.to(device, non_blocking=True)
                    bnd_v = bnd_v.to(device, non_blocking=True)

                    out_v = model(img_v)
                    pl_v, tl_v, bl_v, kl_v, bp_v = (
                        out_v["score_logits"],
                        out_v["thresh_logits"],
                        out_v["boundary_logits"],
                        out_v["k_map"],
                        out_v["binary_map"],
                    )
                    # Score-map
                    prob_score = torch.sigmoid(pl_v)
                    all_pred_probs.append(prob_score.cpu().flatten().numpy())
                    all_gt_labels.append(score_v.cpu().flatten().numpy())
                    # Boundary-map
                    prob_bnd = torch.sigmoid(bl_v)
                    all_pred_probs_bnd.append(prob_bnd.cpu().flatten().numpy())
                    all_gt_labels_bnd.append(bnd_v.cpu().flatten().numpy())
                    # Binary-map
                    prob_bin = torch.sigmoid(bp_v)
                    all_pred_probs_bin.append(prob_bin.cpu().flatten().numpy())
                    all_gt_labels_bin.append(score_v.cpu().flatten().numpy())
                    # K-map distribution
                    all_k_preds.append(kl_v.cpu().flatten().numpy())
                    # Threshold-map errors
                    t_pred = prob_score = torch.sigmoid(tl_v).cpu().flatten().numpy()
                    t_gt = thresh_v.cpu().flatten().numpy()
                    sum_thresh_mse += np.mean((t_pred - t_gt) ** 2)
                    sum_thresh_mae += np.mean(np.abs(t_pred - t_gt))

                    # Existing val losses
                    vLs = hard_negative_mining(
                        nn.functional.binary_cross_entropy_with_logits(
                            pl_v, score_v, reduction="none"
                        ),
                        score_v,
                    )
                    vLb = nn.functional.binary_cross_entropy_with_logits(
                        bp_v, score_v
                    ).mean()
                    vLt = (
                        nn.functional.l1_loss(
                            torch.sigmoid(tl_v), thresh_v, reduction="none"
                        )
                        * (thresh_v > 0)
                    ).sum() / ((thresh_v > 0).sum() + 1e-6)
                    vLbnd = nn.functional.binary_cross_entropy_with_logits(
                        bl_v, bnd_v
                    ).mean()
                    vloss = (
                        vLs + args.alpha * vLb + args.beta * vLt + args.gamma * vLbnd
                    )

                    sum_vLs += vLs.item()
                    sum_vLb += vLb.item()
                    sum_vLt += vLt.item()
                    sum_vLbnd += vLbnd.item()
                    sum_vloss += vloss.item()

                    if i == 0:
                        val_img_raw = transforms.ToPILImage()(
                            inv_normalize(img_v[0].cpu())
                        )

            nvb = len(val_loader)
            # Existing val scalars
            writer.add_scalar("Loss/val_total_epoch", sum_vloss / nvb, ep)
            writer.add_scalar("Loss/val_score_epoch", sum_vLs / nvb, ep)
            writer.add_scalar("Loss/val_binary_epoch", sum_vLb / nvb, ep)
            writer.add_scalar("Loss/val_thresh_epoch", sum_vLt / nvb, ep)
            writer.add_scalar("Loss/val_boundary_epoch", sum_vLbnd / nvb, ep)
            # PR & AUC for score, boundary, binary
            writer.add_pr_curve(
                "PR/val_score",
                np.concatenate(all_gt_labels),
                np.concatenate(all_pred_probs),
                ep,
            )
            writer.add_pr_curve(
                "PR/val_boundary",
                np.concatenate(all_gt_labels_bnd),
                np.concatenate(all_pred_probs_bnd),
                ep,
            )
            writer.add_pr_curve(
                "PR/val_binary",
                np.concatenate(all_gt_labels_bin),
                np.concatenate(all_pred_probs_bin),
                ep,
            )
            # AUC scorers
            for name, gt, pred in [
                ("score", all_gt_labels, all_pred_probs),
                ("boundary", all_gt_labels_bnd, all_pred_probs_bnd),
                ("binary", all_gt_labels_bin, all_pred_probs_bin),
            ]:
                try:
                    auc_val = roc_auc_score(np.concatenate(gt), np.concatenate(pred))
                    writer.add_scalar(f"ROC/val_auc_{name}", auc_val, ep)
                except ValueError:
                    pass
            # Threshold errors
            writer.add_scalar("Metrics/val_thresh_mse", sum_thresh_mse / nvb, ep)
            writer.add_scalar("Metrics/val_thresh_mae", sum_thresh_mae / nvb, ep)
            # K-map distribution
            writer.add_histogram("k_map/val_dist", np.concatenate(all_k_preds), ep)
            # Pixel ratio
            writer.add_scalar("Data/val_pos_pixel_ratio_epoch", pos_ratio_sum / nvb, ep)

            # Save best
            if (avg_val_loss := sum_vloss / nvb) < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth")
                )
                print(
                    f"[Epoch {ep}] New best val loss: {avg_val_loss:.4f}, model saved."
                )

            # Full-image visualization
            prob_full, sc_full, th_full, bd_full = sliding_window_inference(
                val_img_raw,
                model,
                device,
                window_size=args.window_size,
                stride=args.stride,
            )
            writer.add_image("Val/Input_full", transforms.ToTensor()(val_img_raw), ep)
            writer.add_image(
                "Val/Pred_prob_full", torch.from_numpy(prob_full).unsqueeze(0), ep
            )

        scheduler.step()

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
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Минимальный LR для CosineAnnealingLR",
    )
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--shrink_ratio", type=float, default=0.4)
    parser.add_argument("--p_flip", type=float, default=0.5)
    parser.add_argument("--p_rotate", type=float, default=0.1)
    parser.add_argument("--p_color", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=7.0)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args)

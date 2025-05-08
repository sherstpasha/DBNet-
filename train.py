import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR

from dataset import COCOTextDataset
from model import build_model
from utils import hard_negative_mining


def train(args):
    # -------------------------------
    # 1) Select device
    # -------------------------------
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ----------------------
    # 2) Prepare dataset
    # ----------------------
    # Dataset will produce CPU tensors; DataLoader will pin memory
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
        empty_thresh=args.empty_thresh,
    )
    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------------
    # 3) Model, optimizer, scheduler
    # -------------------------------
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

    # ---------------
    # 4) Training loop
    # ---------------
    for ep in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for img_out, score_gt, thresh_gt, bnd_gt in tr_loader:
            # Move batch to GPU (or selected device)
            img_out = img_out.to(device, non_blocking=True)
            score_gt = score_gt.to(device, non_blocking=True)
            thresh_gt = thresh_gt.to(device, non_blocking=True)
            bnd_gt = bnd_gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(img_out)
            pl = outputs["score_logits"]
            tl = outputs["thresh_logits"]
            bl = outputs["boundary_logits"]
            bp = outputs["binary_map"]

            # Compute losses
            Ls = hard_negative_mining(bce(pl, score_gt), score_gt)
            Lb = bce(bp, score_gt).mean()
            Lt = (l1(torch.sigmoid(tl), thresh_gt) * (thresh_gt > 0)).sum() / (
                (thresh_gt > 0).sum() + 1e-6
            )
            Lbnd = bce(bl, bnd_gt).mean()
            loss = Ls + args.alpha * Lb + args.beta * Lt + args.gamma * Lbnd
            print(loss)

            # Backprop
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            epoch_loss += loss.item()
            global_step += 1

        writer.add_scalar("Loss/train_epoch", epoch_loss / len(tr_loader), ep)

        # TODO: add validation loop

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--train_anns", nargs="+", required=True)
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
    parser.add_argument("--empty_thresh", type=float, default=0.01)
    parser.add_argument(
        "--db_k", type=float, default=50.0, help="unused if k_map is learned"
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--device", type=str, default=None, help="torch device string, e.g. 'cuda'"
    )
    parser.add_argument("--log_dir", type=str, default="runs")
    args = parser.parse_args()

    train(args)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tensorboardX import SummaryWriter
from dataset import COCOTextDataset
from utils import hard_negative_mining
from model import build_model
import torch.nn as nn
import numpy as np
import cv2
from torchvision.utils import make_grid, draw_bounding_boxes


def train(
    train_dirs,
    train_anns,
    val_dirs=None,
    val_anns=None,
    epochs=50,
    bs=1,
    lr=1e-4,
    max_lr=1e-3,
    val_split=0.1,
    base_size=1536,
    img_range=256,
    crop_range=(256, 768),
    dynamic_resize=True,
    db_k=50,
    alpha=1.0,
    beta=5.0,
    gamma=1.0,
):
    # --- Датасеты и DataLoader'ы ---
    train_ds = COCOTextDataset(
        train_dirs,
        train_anns,
        base_size=base_size,
        img_range=img_range,
        crop_range=crop_range,
        dynamic_resize=dynamic_resize,
    )
    if val_dirs and val_anns:
        val_ds = COCOTextDataset(
            val_dirs,
            val_anns,
            base_size=base_size,
            img_range=img_range,
            crop_range=None,
            dynamic_resize=dynamic_resize,
        )
    else:
        full_ds = COCOTextDataset(
            train_dirs,
            train_anns,
            base_size=base_size,
            img_range=img_range,
            crop_range=None,
            dynamic_resize=dynamic_resize,
        )
        vn = int(len(full_ds) * val_split)
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - vn, vn])

    trl = DataLoader(train_ds, bs, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds, bs, shuffle=False, num_workers=4, pin_memory=True)

    # --- Модель, оптимизатор, scheduler, лоссы ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(db_k=db_k).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # OneCycleLR: разгон до max_lr за pct_start и спад
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(trl),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=max_lr / lr,
        final_div_factor=1e4,
    )

    bce = nn.BCEWithLogitsLoss(reduction="none")
    l1 = nn.L1Loss(reduction="none")

    writer = SummaryWriter("runs/segdb")
    global_step = 0

    for ep in range(1, epochs + 1):
        # --- Training loop ---
        model.train()
        train_loss = 0.0

        for imgs, sgt, tgt, bgt, _ in trl:
            imgs, sgt, tgt, bgt = [
                t.to(device, non_blocking=True) for t in (imgs, sgt, tgt, bgt)
            ]

            optimizer.zero_grad()
            out = model(imgs)["out"]
            pl, tl, bl = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            pp = torch.sigmoid(pl)
            tp = torch.sigmoid(tl)
            dbp = torch.sigmoid(model.db_k * (pp - tp))

            # Loss components
            Ls = hard_negative_mining(bce(pl, sgt), sgt)
            Lb = bce(dbp, sgt).mean()
            Lt = (l1(tp, tgt) * (tgt > 0)).sum() / ((tgt > 0).sum() + 1e-6)
            Lbnd = bce(bl, bgt).mean()

            loss = Ls + alpha * Lb + beta * Lt + gamma * Lbnd
            loss.backward()
            optimizer.step()
            scheduler.step()  # обновляем lr каждый шаг

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1
            train_loss += loss.item()

        avg_train_loss = train_loss / len(trl)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, ep)

        # --- Validation loop ---
        model.eval()
        val_loss = 0.0
        saved_examples = None

        with torch.no_grad():
            for imgs, sgt, tgt, bgt, raw in vl:
                imgs, sgt, tgt, bgt = [
                    t.to(device, non_blocking=True) for t in (imgs, sgt, tgt, bgt)
                ]
                out = model(imgs)["out"]
                pl, tl, bl = out[:, 0:1], out[:, 1:2], out[:, 2:3]

                pp = torch.sigmoid(pl)
                tp = torch.sigmoid(tl)
                dbp = torch.sigmoid(model.db_k * (pp - tp))

                Ls = hard_negative_mining(bce(pl, sgt), sgt)
                Lb = bce(dbp, sgt).mean()
                Lt = (l1(tp, tgt) * (tgt > 0)).sum() / ((tgt > 0).sum() + 1e-6)
                Lbnd = bce(bl, bgt).mean()

                val_loss += (Ls + alpha * Lb + beta * Lt + gamma * Lbnd).item()

                # сохраним один батч для логов
                if saved_examples is None:
                    saved_examples = (raw, sgt, tgt, dbp, tp, bl)

        avg_val_loss = val_loss / len(vl)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, ep)

        # --- Логирование карт и боксов каждые 5 эпох ---
        if ep % 5 == 0 and saved_examples is not None:
            raw, sgt, tgt, dbp, tp, bl = saved_examples

            # GT карты
            writer.add_image("Val/GT_Static", make_grid(sgt, normalize=True), ep)
            writer.add_image("Val/GT_Thresh", make_grid(tgt, normalize=True), ep)
            writer.add_image("Val/GT_Boundary", make_grid(bgt, normalize=True), ep)

            # Предсказания
            writer.add_image("Val/Pred_Static", make_grid(dbp, normalize=True), ep)
            writer.add_image("Val/Pred_Thresh", make_grid(tp, normalize=True), ep)
            writer.add_image(
                "Val/Pred_Boundary", make_grid(torch.sigmoid(bl), normalize=True), ep
            )

            # Боксы на dilated GT
            gt_mask = sgt[0, 0].cpu().numpy().astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gt_dil = cv2.dilate(gt_mask, kernel, iterations=1)
            cnts_gt, _ = cv2.findContours(
                gt_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts_gt:
                boxes_gt = torch.tensor(
                    [
                        [x, y, x + w, y + h]
                        for x, y, w, h in map(cv2.boundingRect, cnts_gt)
                    ],
                    dtype=torch.int,
                )
                img_gt = draw_bounding_boxes(raw[0], boxes_gt, colors="red", width=2)
                writer.add_image("Val/Boxes_GT_Dilated", img_gt, ep)

            # Боксы на предсказании static
            pred_mask = (dbp > 0.5)[0, 0].cpu().numpy().astype(np.uint8)
            cnts_p, _ = cv2.findContours(
                pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts_p:
                boxes_p = torch.tensor(
                    [
                        [x, y, x + w, y + h]
                        for x, y, w, h in map(cv2.boundingRect, cnts_p)
                    ],
                    dtype=torch.int,
                )
                img_pr = draw_bounding_boxes(raw[0], boxes_p, colors="green", width=2)
                writer.add_image("Val/Boxes_Pred", img_pr, ep)

        # --- Сохраняем чекпоинт ---
        torch.save(model.state_dict(), f"checkpoint_epoch{ep}.pth")

    writer.close()

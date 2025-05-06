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
    # --- Prepare datasets and loaders ---
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

    # --- Model, optimizer, scheduler, losses ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(db_k=db_k).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        # --- Training ---
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

            Ls = hard_negative_mining(bce(pl, sgt), sgt)
            Lb = bce(dbp, sgt).mean()
            Lt = (l1(tp, tgt) * (tgt > 0)).sum() / ((tgt > 0).sum() + 1e-6)
            Lbnd = bce(bl, bgt).mean()

            loss = Ls + alpha * Lb + beta * Lt + gamma * Lbnd
            loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
            train_loss += loss.item()

        avg_train = train_loss / len(trl)
        writer.add_scalar("Loss/train_epoch", avg_train, ep)

        # --- Validation loop ---
        model.eval()
        val_loss = 0.0
        saved = None

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

                # Loss components
                Ls = hard_negative_mining(bce(pl, sgt), sgt)
                Lb = bce(dbp, sgt).mean()
                Lt = (l1(tp, tgt) * (tgt > 0)).sum() / ((tgt > 0).sum() + 1e-6)
                Lbnd = bce(bl, bgt).mean()

                # Собираем общий лосс и прибавляем
                loss_val = Ls + alpha * Lb + beta * Lt + gamma * Lbnd
                val_loss += loss_val.item()

                # Сохраним первый батч для логов
                if saved is None:
                    saved = {
                        "raw": raw[0].cpu(),
                        "gt_s": sgt[0, 0].cpu(),
                        "gt_b": bgt[0, 0].cpu(),
                        "pr_s": dbp[0, 0].cpu(),
                        "pr_b": torch.sigmoid(bl)[0, 0].cpu(),
                    }

        avg_val = val_loss / len(vl)
        writer.add_scalar("Loss/val_epoch", avg_val, ep)

        # … внутри цикла по эпохам, сразу после расчёта avg_val …
        if ep % 1 == 0 and saved is not None:
            # === достаём всё из saved ===
            raw_img = (
                saved["raw"].permute(1, 2, 0).numpy().astype(np.uint8)
            )  # H×W×3, RGB
            gt_static = (saved["gt_s"].numpy() * 255).astype(np.uint8)  # H×W
            gt_boundary = (saved["gt_b"].numpy() * 255).astype(np.uint8)  # H×W
            pr_static = (saved["pr_s"].numpy() * 255).astype(np.uint8)  # H×W
            pr_boundary = (saved["pr_b"].numpy() * 255).astype(np.uint8)  # H×W

            # === вспомогательная функция оверлея ===
            def make_overlay(static, boundary, color_idx):
                # static, boundary — H×W uint8
                static_bgr = cv2.cvtColor(static, cv2.COLOR_GRAY2BGR)
                bnd_bgr = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
                # красим маску границ в нужный канал (2=R,1=G,0=B)
                bnd_bgr[:, :, color_idx] = boundary
                # blend
                return cv2.addWeighted(static_bgr, 0.7, bnd_bgr, 0.3, 0)

            # === делаем оверлеи ===
            gt_ov = make_overlay(gt_static, gt_boundary, color_idx=2)  # красный
            pr_ov = make_overlay(pr_static, pr_boundary, color_idx=1)  # зелёный

            # === конвертируем оверлеи в RGB (TensorBoard любит RGB) ===
            gt_rgb = cv2.cvtColor(gt_ov, cv2.COLOR_BGR2RGB)
            pr_rgb = cv2.cvtColor(pr_ov, cv2.COLOR_BGR2RGB)
            # raw_img у нас уже RGB, можно сразу использовать

            # === собираем компаративное изображение: raw | GT | pred ===
            comp = np.concatenate([raw_img, gt_rgb, pr_rgb], axis=1)  # H×(3W)×3

            # === приводим к [C×H×W] для writer.add_image ===
            comp_t = torch.from_numpy(comp).permute(2, 0, 1)

            writer.add_image("Val/Comparison", comp_t, ep)

        # --- Save checkpoint ---
        torch.save(model.state_dict(), f"checkpoint_epoch{ep}.pth")

    writer.close()

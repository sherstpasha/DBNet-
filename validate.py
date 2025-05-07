import os
import argparse
import random
import csv

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from dataset import COCOTextDataset
from model import build_model
from utils import sliding_window_inference, match_detections


def preds_to_boxes(prob_map, thresh, min_area, expand_ratio, img_w, img_h):
    """
    Из probability map в список bbox [x1,y1,x2,y2], с расширением каждого бокса на expand_ratio.
    """
    mask = (prob_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        # исходный бокс
        x1, y1, x2, y2 = x, y, x + w, y + h
        # расширяем
        dw = w * expand_ratio / 2
        dh = h * expand_ratio / 2
        nx1 = max(0, int(x1 - dw))
        ny1 = max(0, int(y1 - dh))
        nx2 = min(img_w, int(x2 + dw))
        ny2 = min(img_h, int(y2 + dh))
        boxes.append([nx1, ny1, nx2, ny2])
    return boxes

def evaluate(args):
    # 1) Модель
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(db_k=args.db_k).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 2) Датасет
    ds = COCOTextDataset(
        images_dirs=args.test_dirs,
        ann_files=args.test_anns,
        base_size=args.base_size,
        crop_range=None,
        p_flip=0, p_rotate=0, p_color=0,
        max_angle=0, empty_thresh=0,
    )
    # подвыборка
    if args.subset_frac < 1.0:
        n = max(1, int(len(ds) * args.subset_frac))
        idxs = random.sample(range(len(ds)), n)
        ds = Subset(ds, idxs)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    # 3) Сетка параметров
    det_threshes = np.arange(args.det_thresh_start,
                             args.det_thresh_stop + 1e-9,
                             args.det_thresh_step)
    min_areas    = np.arange(args.min_area_start,
                             args.min_area_stop + 1,
                             args.min_area_step, dtype=int)

    # 4) Инициализация метрик
    metrics = {(dt, ma): {"tp": 0, "fp": 0, "fn": 0}
               for dt in det_threshes for ma in min_areas}

    # 5) Пробег по изображениям
    for img_t, score_t, thresh_t, bnd_t, raw, gt_boxes in loader:
        raw_np = raw[0].permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(raw_np)

        # получаем четыре карты, но работаем только с prob_map
        prob_map, _, _, _ = sliding_window_inference(
            pil, model, device,
            window_size=args.window_size,
            stride=args.stride,
            db_k=args.db_k
        )

        gts = gt_boxes[0].cpu().tolist()
        h, w = raw_np.shape[:2]

        for dt in det_threshes:
            for ma in min_areas:
                preds = preds_to_boxes(
                    prob_map, thresh=dt, min_area=ma,
                    expand_ratio=args.box_expand,
                    img_w=w, img_h=h
                )
                tp, fp, fn = match_detections(preds, gts, args.iou_thresh)
                m = metrics[(dt, ma)]
                m["tp"] += tp; m["fp"] += fp; m["fn"] += fn

    # 6) Запись CSV и поиск лучшего
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    best = (None, None, -1.0)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["det_thresh", "min_area", "precision", "recall", "f1"])
        for dt in det_threshes:
            for ma in min_areas:
                tp = metrics[(dt, ma)]["tp"]
                fp = metrics[(dt, ma)]["fp"]
                fn = metrics[(dt, ma)]["fn"]
                prec = tp / (tp + fp + 1e-6)
                rec  = tp / (tp + fn + 1e-6)
                f1   = 2 * prec * rec / (prec + rec + 1e-6)
                writer.writerow([dt, ma, prec, rec, f1])
                if f1 > best[2]:
                    best = (dt, ma, f1)
    print(f"Done, results -> {args.output_csv}")
    print(f"Best: det_thresh={best[0]:.2f}, min_area={best[1]}, F1={best[2]:.4f}")

    # 7) Сохранение визуализаций лучших
    os.makedirs(args.vis_dir, exist_ok=True)
    best_dt, best_ma, _ = best

    for i in range(min(args.vis_n, len(ds))):
        # Получаем изображение и боксы
        img_t, score_t, thresh_t, bnd_t, raw, gt_boxes = ds[i]
        raw_np = raw.permute(1,2,0).numpy()
        pil = Image.fromarray(raw_np)

        # Тiled inference для получения prob_map
        prob_map, _, _, _ = sliding_window_inference(
            pil, model, device,
            window_size=args.window_size,
            stride=args.stride,
            db_k=args.db_k
        )

        # Генерируем предикт-боксы и GT-боксы
        preds = preds_to_boxes(
            prob_map, best_dt, best_ma,
            expand_ratio=args.box_expand,
            img_w=raw_np.shape[1], img_h=raw_np.shape[0]
        )
        gts = gt_boxes.tolist()

        # Рисуем боксы на копии изображения
        vis = raw_np.copy()

        overlay = vis.copy()
        # рисуем прямоугольники на overlay
        for x1, y1, x2, y2 in gts:
            cv2.rectangle(overlay,
                        (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 0, 255), 2)  # GT — красным
        for x1, y1, x2, y2 in preds:
            cv2.rectangle(overlay,
                        (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 2)  # Pred — зелёным
                    
        # задаём прозрачность
        alpha = 0.8

        # накладываем overlay на оригинал
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)            

        # Масштабируем: большая сторона = 1024px
        h0, w0 = vis.shape[:2]
        scale = 1024.0 / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        vis_resized = cv2.resize(vis, (new_w, new_h))

        # Добавляем полосу-легенду внизу (40px)
        legend_h = 40
        canvas = np.ones((new_h + legend_h, new_w, 3), dtype=np.uint8) * 255
        canvas[:new_h, :new_w] = vis_resized

        # Красный квадратик + подпись GT
        cv2.rectangle(canvas,
                    (10, new_h+5), (30, new_h+25),
                    (0,0,255), -1)
        cv2.putText(canvas, "Ground truth (Blue)",
                    (35, new_h+22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 1, cv2.LINE_AA)

        # Зелёный квадратик + подпись Predict
        green_x = 260  # новая стартовая позиция по X
        cv2.rectangle(canvas,
                    (green_x, new_h+5), (green_x+20, new_h+25),
                    (0,255,0), -1)
        cv2.putText(canvas, "Predict (Green)",
                    (green_x+25, new_h+22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 1, cv2.LINE_AA)
        
        # Сохраняем итоговый PNG
        out_path = os.path.join(
            args.vis_dir,
            f"vis_{i:02d}_dt{best_dt:.2f}_ma{best_ma}.png"
        )
        cv2.imwrite(out_path,
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt",        type=str,   required=True)
    parser.add_argument("--test_dirs",         nargs="+", required=True)
    parser.add_argument("--test_anns",         nargs="+", required=True)
    parser.add_argument("--db_k",              type=float, default=50.0)
    parser.add_argument("--window_size",       type=int,   default=512)
    parser.add_argument("--stride",            type=int,   default=256)
    parser.add_argument("--iou_thresh",        type=float, default=0.5)
    parser.add_argument("--base_size",         type=int,   default=2048)

    parser.add_argument("--det_thresh_start",  type=float, default=0.1)
    parser.add_argument("--det_thresh_stop",   type=float, default=0.9)
    parser.add_argument("--det_thresh_step",   type=float, default=0.1)

    parser.add_argument("--min_area_start",    type=int,   default=5)
    parser.add_argument("--min_area_stop",     type=int,   default=100)
    parser.add_argument("--min_area_step",     type=int,   default=5)

    parser.add_argument("--subset_frac",       type=float, default=1.0,
                        help="доля тестовой выборки (0.1 = 10%%)")
    parser.add_argument("--box_expand",        type=float, default=0.3,
                        help="коэффициент расширения предикт-боксов (доля от размера бокса)")
    parser.add_argument("--vis_dir",           type=str,   default="vis",
                        help="куда сохранять PNG-визуализации")
    parser.add_argument("--vis_n",             type=int,   default=5,
                        help="сколько примеров визуализировать")
    parser.add_argument("--output_csv",        type=str,   default="validation_results.csv")
    parser.add_argument("--device",            type=str,   default="cuda")
    args = parser.parse_args()

    evaluate(args)

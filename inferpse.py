import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

from model import build_model
from utils import sliding_window_inference


def resize_to_base(image: Image.Image, base_size: int) -> Image.Image:
    w, h = image.size
    scale = base_size / max(w, h)
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR)
    return image


def instance_segmentation_via_thresholds(
    prob_map: np.ndarray, bnd_map: np.ndarray, high_th: float = 0.8, low_th: float = 0.6
) -> np.ndarray:
    """
    Делим на инстансы так:
     1) seeds = prob_map >= high_th
     2) mask  = prob_map >= low_th
     3) markers = connectedComponents(seeds)
     4) watershed на градиенте prob_map внутри mask
    Возвращает карту меток той же формы, где 0=фон, 1,2,...=разные регионы.
    """
    seeds = (prob_map >= high_th).astype(np.uint8)
    n_seeds, markers = cv2.connectedComponents(seeds)

    # смещаем метки, чтобы фон был 0, заливки >0
    markers = markers + 1
    markers[seeds == 0] = 0

    mask = (prob_map >= low_th).astype(np.uint8)
    grad = cv2.morphologyEx(
        (prob_map * 255).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # watershed
    img3 = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img3, markers.copy())

    inst = markers_ws.copy()
    inst[inst == -1] = 0
    inst[mask == 0] = 0
    return inst


def overlay_instances(img: Image.Image, inst_map: np.ndarray) -> Image.Image:
    """
    Накладывает на каждый инстанс полупрозрачный цвет.
    """
    h, w = inst_map.shape
    base = img.convert("RGBA").resize((w, h))
    np_overlay = np.zeros((h, w, 4), np.uint8)

    n_labels = inst_map.max()
    # случайные цвета
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(n_labels + 1)]
    for lbl in range(1, n_labels + 1):
        mask = inst_map == lbl
        c = colors[lbl]
        alpha = 100
        np_overlay[mask] = (c[0], c[1], c[2], alpha)

    overlay = Image.fromarray(np_overlay, mode="RGBA")
    return Image.alpha_composite(base, overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--base_size", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--high_th", type=float, default=0.9)
    parser.add_argument("--low_th", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1) Модель
    model = build_model().to(device)
    sd = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # 2) Изображение
    img_path = Path(args.image_path)
    img = Image.open(img_path).convert("RGB")
    img = resize_to_base(img, args.base_size)

    # 3) Скользящий inference
    prob_map, score_map, thresh_map, bnd_map = sliding_window_inference(
        img,
        model,
        device,
        window_size=args.window_size,
        stride=args.stride,
    )

    # 4) Инстанс-сегментация
    inst_map = instance_segmentation_via_thresholds(
        prob_map,
        bnd_map,
        high_th=args.high_th,
        low_th=args.low_th,
    )

    # 5) Оверлей и показ
    out = overlay_instances(img, inst_map)
    out.show()

    # 6) Сохранение
    save_path = img_path.with_name(img_path.stem + "_inferInst" + img_path.suffix)
    out.convert("RGB").save(save_path)
    print(f"Saved instance-overlay to {save_path}")


if __name__ == "__main__":
    main()

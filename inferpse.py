import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
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
):
    """
    Делим на инстансы так:
     1) seeds = prob_map >= high_th
     2) mask = prob_map >= low_th
     3) markers = connectedComponents(seeds)
     4) watershed на градиенте prob_map внутри mask
    Возвращает карту меток той же формы, где 0=фон, 1,2,...=разные регионы.
    """
    # 1) seeds
    seeds = (prob_map >= high_th).astype(np.uint8)
    n_seeds, markers = cv2.connectedComponents(seeds)

    # смещаем метки фона в 1, а всё остальное +1 (чтоб 0 оставался «невизначен» для watershed)
    markers = markers + 1
    markers[seeds == 0] = 0

    # 2) общий mask
    mask = (prob_map >= low_th).astype(np.uint8)

    # 3) делаем градиент от prob_map
    grad = cv2.morphologyEx(
        (prob_map * 255).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # 4) watershed
    # преобразуем mask в трёхканальное изображение для cv2.watershed
    img3 = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img3, markers.copy())

    # после watershed границы отмечены -1, затираем их
    inst = markers_ws.copy()
    inst[inst == -1] = 0
    # ограничиваем область зоной mask
    inst[mask == 0] = 0
    return inst


def overlay_instances(img: Image.Image, inst_map: np.ndarray):
    """
    На каждый инстанс накладываем полупрозрачный цвет.
    """
    h, w = inst_map.shape
    base = img.convert("RGBA").resize((w, h))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    np_overlay = np.zeros((h, w, 4), np.uint8)

    # сгенерим случайные цвета для каждого инстанса
    n_labels = inst_map.max()
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(n_labels + 1)]
    for lbl in range(1, n_labels + 1):
        mask = inst_map == lbl
        c = colors[lbl]
        alpha = 100  # полупрозрачность
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

    model = build_model().to(device)
    sd = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    img = Image.open(args.image_path).convert("RGB")
    img = resize_to_base(img, args.base_size)

    prob_map, score_map, thresh_map, bnd_map = sliding_window_inference(
        img, model, device, window_size=args.window_size, stride=args.stride
    )

    inst_map = instance_segmentation_via_thresholds(
        prob_map, bnd_map, high_th=args.high_th, low_th=args.low_th
    )

    overlaid = overlay_instances(img, inst_map)
    overlaid.show()


if __name__ == "__main__":
    main()

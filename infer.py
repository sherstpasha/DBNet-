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
    """Масштабирует PIL-изображение так, чтобы большая сторона стала base_size."""
    w, h = image.size
    scale = base_size / max(w, h)
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR)
    return image


def overlay_mask_and_contours(
    img: Image.Image, sep_mask: np.ndarray, alpha: float = 0.4, line_width: int = 2
) -> Image.Image:
    """
    Накладывает полупрозрачную красную маску `sep_mask` на PIL-изображение
    и обводит найденные контуры чёрной линией.
    """
    # 1) полупрозрачная заливка
    bin_mask = (sep_mask > 0).astype(np.uint8) * 255
    alpha_mask = Image.fromarray((bin_mask * alpha).astype(np.uint8), mode="L")
    red_layer = Image.new("RGBA", img.size, (255, 0, 0, 0))
    red_layer.putalpha(alpha_mask)
    base = img.convert("RGBA")
    overlaid = Image.alpha_composite(base, red_layer)

    # 2) поиск контуров и обводка
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(overlaid)
    for cnt in contours:
        # cnt — это N×1×2 array; преобразуем в list of tuples
        pts = cnt.reshape(-1, 2)
        polygon = [tuple(pt) for pt in pts]
        draw.line(polygon + [polygon[0]], fill="black", width=line_width)

    return overlaid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="path to image")
    parser.add_argument("--ckpt_path", required=True, help="path to .pth checkpoint")
    parser.add_argument(
        "--base_size",
        type=int,
        default=2048,
        help="какой размер выставить большей стороне перед inference",
    )
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument(
        "--prob_th", type=float, default=0.5, help="порог для основной маски"
    )
    parser.add_argument(
        "--bnd_th", type=float, default=0.5, help="порог для маски границ"
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="прозрачность заливки")
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

    # 2) Загружаем и ресайзим
    img = Image.open(args.image_path).convert("RGB")
    img = resize_to_base(img, args.base_size)

    # 3) Скользящий inference
    prob_map, score_map, thresh_map, bnd_map = sliding_window_inference(
        img,
        model,
        device,
        window_size=args.window_size,
        stride=args.stride,
    )

    # 4) Пост-процессинг: пороги и вычитание границ
    mask_prob = (prob_map >= args.prob_th).astype(np.uint8)
    mask_bnd = (bnd_map >= args.bnd_th).astype(np.uint8)
    sep_mask = cv2.subtract(mask_prob, mask_bnd)

    # 5) Overlay + контуры → показываем
    out = overlay_mask_and_contours(img, sep_mask, alpha=args.alpha)
    out.show()


if __name__ == "__main__":
    main()

from pathlib import Path
from dataset import COCOTextDataset
import random
import torch
from PIL import Image, ImageDraw

# 1) Создаём датасет
dataset = COCOTextDataset(
    images_dirs=r"C:\shared\data0205\Archives020525\test_images",
    ann_files=r"C:\shared\data0205\Archives020525\test.json",
    base_size=4096,
    crop_range=(512, 1024),
    p_flip=0.1,
    p_rotate=0.2,       # чуть больше рандома, если нужно
    max_angle=7,
    p_color=0.1,
)

# 2) Берём несколько примеров и показываем с таргет-боксами
for _ in range(3):  # пробуем 3 примера
    img_t, score_t, thresh_t, boundary_t, raw_img, boxes_t = dataset[random.randrange(len(dataset))]
    
    # raw_img: Tensor[C,H,W] uint8 → PIL
    raw_np = raw_img.permute(1, 2, 0).cpu().numpy()
    raw_pil = Image.fromarray(raw_np)

    # Рисуем боксы на копии
    draw = ImageDraw.Draw(raw_pil)
    for box in boxes_t:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    raw_pil.show(title="Raw with GT boxes")
    
    # score map → серый PIL
    score_np = (score_t.squeeze(0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(score_np).show(title="Score map")

    # threshold map → серый PIL
    thresh_np = (thresh_t.squeeze(0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(thresh_np).show(title="Threshold map")

    # boundary map → серый PIL
    bound_np = (boundary_t.squeeze(0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(bound_np).show(title="Boundary map")

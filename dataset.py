import json
import random
from pathlib import Path
from itertools import zip_longest

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class COCOTextDataset(Dataset):
    def __init__(
        self,
        images_dirs,
        ann_files,
        base_size=1536,
        img_range=256,
        crop_range=(256, 768),
        shrink_ratio=0.3,
        p_flip=0.1,
        p_rotate=0.1,
        p_color=0.1,
        max_angle=15,
        dynamic_resize=True,
    ):
        # Normalize inputs to lists
        if isinstance(images_dirs, (str, Path)):
            images_dirs = [images_dirs]
        self.images_dirs = [Path(p) for p in images_dirs]
        if isinstance(ann_files, (str, Path)):
            ann_files = [ann_files]
        self.ann_files = ann_files

        # Augmentation and resize parameters
        self.base_size = base_size
        self.img_range = img_range
        self.crop_range = crop_range
        self.shrink_ratio = shrink_ratio
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_color = p_color
        self.max_angle = max_angle
        self.dynamic_resize = dynamic_resize

        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load COCO-format annotations
        self.images = {}
        self.anns = {}
        self.img_map = {}
        for ann_file, img_dir in zip_longest(
            self.ann_files, self.images_dirs, fillvalue=self.images_dirs[-1]
        ):
            data = json.load(open(ann_file, "r", encoding="utf8"))
            for im in data.get("images", []):
                img_id = im["id"]
                self.images[img_id] = im
                self.img_map[img_id] = Path(img_dir)
            for ann in data.get("annotations", []):
                self.anns.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        # Load and convert image
        image = Image.open(self.img_map[img_id] / info["file_name"]).convert("RGB")
        orig_w, orig_h = image.size

        # Prepare shrunk and dilated masks
        shrunk_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        dilated_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in self.anns.get(img_id, []):
            poly = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2)
            center = poly.mean(axis=0)
            shr = center + (poly - center) * (1 - self.shrink_ratio)
            dil = center + (poly - center) * (1 + self.shrink_ratio)
            cv2.fillPoly(shrunk_mask, [shr.astype(np.int32)], 1)
            cv2.fillPoly(dilated_mask, [dil.astype(np.int32)], 1)

        # Distance transform for threshold map
        d1 = cv2.distanceTransform((shrunk_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        d2 = cv2.distanceTransform((dilated_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        thresh_full = (d1 / (d1 + d2 + 1e-6)) * dilated_mask
        score_full = shrunk_mask

        # Boundary map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        text_mask = (score_full > 0).astype(np.uint8)
        dil_text = cv2.dilate(text_mask, kernel, iterations=2)
        boundary_full = np.clip(dil_text - text_mask, 0, 1).astype(np.uint8)

        # Data augmentations: flip, rotate, color jitter
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            score_full = np.fliplr(score_full)
            thresh_full = np.fliplr(thresh_full)
            boundary_full = np.fliplr(boundary_full)
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = image.rotate(angle, resample=Image.BILINEAR)
            M = cv2.getRotationMatrix2D((orig_w / 2, orig_h / 2), angle, 1)
            score_full = cv2.warpAffine(
                score_full, M, (orig_w, orig_h), flags=cv2.INTER_NEAREST
            )
            thresh_full = (
                cv2.warpAffine(
                    (thresh_full * 255).astype(np.uint8),
                    M,
                    (orig_w, orig_h),
                    flags=cv2.INTER_LINEAR,
                )
                / 255.0
            )
            boundary_full = cv2.warpAffine(
                boundary_full, M, (orig_w, orig_h), flags=cv2.INTER_NEAREST
            )
        if random.random() < self.p_color:
            image = self.color_jitter(image)

        # Resize: dynamic or keep original
        if self.dynamic_resize:
            out_w = out_h = random.randint(
                self.base_size - self.img_range, self.base_size + self.img_range
            )
        else:
            out_w, out_h = orig_w, orig_h
        image = image.resize((out_w, out_h), Image.BILINEAR)
        score_full = cv2.resize(
            score_full, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        thresh_full = (
            cv2.resize(
                (thresh_full * 255).astype(np.uint8),
                (out_w, out_h),
                interpolation=cv2.INTER_LINEAR,
            )
            / 255.0
        )
        boundary_full = cv2.resize(
            boundary_full, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )

        # Random crop
        if self.crop_range:
            min_c, max_c = self.crop_range
            patch = random.randint(min_c, max_c)
            if patch < out_w and patch < out_h:
                left = random.randint(0, out_w - patch)
                top = random.randint(0, out_h - patch)
                image = image.crop((left, top, left + patch, top + patch))
                score_full = score_full[top : top + patch, left : left + patch]
                thresh_full = thresh_full[top : top + patch, left : left + patch]
                boundary_full = boundary_full[top : top + patch, left : left + patch]

        # To tensors
        img_t = self.normalize(image)
        score_t = torch.from_numpy((score_full > 0).astype(np.float32)).unsqueeze(0)
        thresh_t = torch.from_numpy(thresh_full.astype(np.float32)).unsqueeze(0)
        boundary_t = torch.from_numpy(boundary_full.astype(np.float32)).unsqueeze(0)
        raw_img = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.uint8)

        return img_t, score_t, thresh_t, boundary_t, raw_img

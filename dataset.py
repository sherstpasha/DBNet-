import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia.augmentation as K
import torch.nn.functional as F


class COCOTextDataset(Dataset):
    def __init__(
        self,
        images_dirs,
        ann_files,
        base_size=4096,
        crop_size=512,
        shrink_ratio=0.3,
        p_flip=0.1,
        p_rotate=0.1,
        p_color=0.1,
        max_angle=7,
        empty_thresh=0.01,
    ):
        # Normalize paths
        if isinstance(images_dirs, (str, Path)):
            images_dirs = [images_dirs]
        self.images_dirs = [Path(p) for p in images_dirs]
        if isinstance(ann_files, (str, Path)):
            ann_files = [ann_files]
        self.ann_files = ann_files

        # Parameters
        self.base_size = base_size
        self.crop_size = crop_size
        self.shrink_ratio = shrink_ratio
        self.empty_thresh = empty_thresh

        # Transforms (all on CPU)
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.augment = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=p_flip),
            K.RandomRotation(degrees=max_angle, p=p_rotate),
            K.RandomCrop(size=(crop_size, crop_size), p=1.0),
        )
        self.color_augment = K.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p_color
        )

        # Load annotations
        self.images = {}
        self.anns = {}
        self.img_map = {}
        for ann_file, img_dir in zip(self.ann_files, self.images_dirs):
            data = json.load(open(ann_file, "r", encoding="utf8"))
            for im in data.get("images", []):
                img_id = im["id"]
                self.images[img_id] = im
                self.img_map[img_id] = img_dir
            for ann in data.get("annotations", []):
                img_id = ann["image_id"]
                self.anns.setdefault(img_id, []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def _resize_if_needed(self, image: Image.Image):
        orig_w, orig_h = image.size
        scale = self.base_size / max(orig_w, orig_h)
        if scale != 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            return image, scale, new_w, new_h
        return image, scale, orig_w, orig_h

    def _build_masks(self, img_id, scale, H, W):
        # Build shrunk and dilated binary masks on CPU
        shr = np.zeros((H, W), dtype=np.uint8)
        dil = np.zeros((H, W), dtype=np.uint8)
        for ann in self.anns.get(img_id, []):
            coords = (
                np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2)
                * scale
            )
            center = coords.mean(axis=0)
            shr_poly = center + (coords - center) * (1 - self.shrink_ratio)
            dil_poly = center + (coords - center) * (1 + self.shrink_ratio)
            cv2.fillPoly(shr, [shr_poly.astype(np.int32)], 1)
            cv2.fillPoly(dil, [dil_poly.astype(np.int32)], 1)
        # Convert to CPU tensors
        shr_t = torch.from_numpy(shr).unsqueeze(0).unsqueeze(0).float()
        dil_t = torch.from_numpy(dil).unsqueeze(0).unsqueeze(0).float()
        return shr_t, dil_t

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]

        # Load image and static resize on CPU
        image = Image.open(self.img_map[img_id] / info["file_name"]).convert("RGB")
        image, scale, orig_w, orig_h = self._resize_if_needed(image)
        H, W = orig_h, orig_w

        # Build masks on CPU
        shr_t, dil_t = self._build_masks(img_id, scale, H, W)

        # Distance transform via OpenCV on CPU
        shr_np = shr_t.squeeze(0).squeeze(0).numpy().astype(np.uint8)
        dil_np = dil_t.squeeze(0).squeeze(0).numpy().astype(np.uint8)
        d1_np = cv2.distanceTransform(shr_np, cv2.DIST_L2, 5)
        d2_np = cv2.distanceTransform(dil_np, cv2.DIST_L2, 5)
        d1_t = torch.from_numpy(d1_np).unsqueeze(0).unsqueeze(0).float()
        d2_t = torch.from_numpy(d2_np).unsqueeze(0).unsqueeze(0).float()

        # Compute targets on CPU
        score_t = shr_t
        thresh_t = d1_t / (d1_t + d2_t + 1e-6) * dil_t
        bnd_t = (
            F.max_pool2d(score_t, kernel_size=3, stride=1, padding=1) - score_t
        ).clamp(min=0)

        # Combine and augment on CPU
        img_t = TF.to_tensor(image).unsqueeze(0)
        combined = torch.cat([img_t, score_t, thresh_t, bnd_t], dim=1)
        aug = self.augment(combined)
        img_t, score_t, thresh_t, bnd_t = (
            aug[:, :3],
            aug[:, 3:4],
            aug[:, 4:5],
            aug[:, 5:6],
        )
        img_t = self.color_augment(img_t)

        # Final normalize on CPU
        img_out = self.normalize(TF.to_pil_image(img_t.squeeze(0)))

        return img_out, score_t.squeeze(0), thresh_t.squeeze(0), bnd_t.squeeze(0)

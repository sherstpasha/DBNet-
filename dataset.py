import json
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class COCOTextDataset(Dataset):
    def __init__(
        self,
        images_dirs,
        ann_files,
        base_size=4096,
        crop_range=(512, 1024),
        shrink_ratio=0.3,
        p_flip=0.1,
        p_rotate=0.1,
        p_color=0.1,
        max_angle=7,
        empty_thresh=0.01,
    ):
        # Normalize inputs to lists
        if isinstance(images_dirs, (str, Path)):
            images_dirs = [images_dirs]
        self.images_dirs = [Path(p) for p in images_dirs]
        if isinstance(ann_files, (str, Path)):
            ann_files = [ann_files]
        self.ann_files = ann_files

        # Resize and augmentation parameters
        self.base_size = base_size
        self.crop_range = crop_range
        self.shrink_ratio = shrink_ratio
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_color = p_color
        self.max_angle = max_angle
        self.empty_thresh = empty_thresh

        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load COCO-format annotations: polygons and bboxes
        self.images = {}
        self.anns = {}
        self.boxes = {}
        self.img_map = {}
        for ann_file, img_dir in zip(self.ann_files, self.images_dirs):
            data = json.load(open(ann_file, "r", encoding="utf8"))
            for im in data.get("images", []):
                img_id = im["id"]
                self.images[img_id] = im
                self.img_map[img_id] = Path(img_dir)
            for ann in data.get("annotations", []):
                img_id = ann["image_id"]
                self.anns.setdefault(img_id, []).append(ann)
                # store original bbox [x1,y1,x2,y2]
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                self.boxes.setdefault(img_id, []).append([x, y, x + w, y + h])
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        # 1) Load and (if needed) static resize
        image = Image.open(self.img_map[img_id] / info["file_name"]).convert("RGB")
        orig_w, orig_h = image.size
        scale = 1.0
        if max(orig_w, orig_h) > self.base_size:
            scale = self.base_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            orig_w, orig_h = new_w, new_h

        # 2) Build shrunk / dilated masks for DBNet targets
        shrunk_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        dilated_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in self.anns.get(img_id, []):
            poly = (np.array(ann["segmentation"][0], dtype=np.float32)
                    .reshape(-1,2) * scale)
            center = poly.mean(axis=0)
            shr = center + (poly - center) * (1 - self.shrink_ratio)
            dil = center + (poly - center) * (1 + self.shrink_ratio)
            cv2.fillPoly(shrunk_mask, [shr.astype(np.int32)], 1)
            cv2.fillPoly(dilated_mask, [dil.astype(np.int32)], 1)

        d1 = cv2.distanceTransform((shrunk_mask==0).astype(np.uint8), cv2.DIST_L2, 5)
        d2 = cv2.distanceTransform((dilated_mask==0).astype(np.uint8), cv2.DIST_L2, 5)
        thresh_full = (d1 / (d1 + d2 + 1e-6)) * dilated_mask
        score_full  = shrunk_mask.copy()

        # 3) Boundary target
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        text_mask = (score_full>0).astype(np.uint8)
        dil_text = cv2.dilate(text_mask, kernel, iterations=2)
        boundary_full = np.clip(dil_text - text_mask, 0, 1).astype(np.uint8)

        # 4) Augmentations
        if random.random() < self.p_flip:
            image       = TF.hflip(image)
            score_full  = np.fliplr(score_full)
            thresh_full = np.fliplr(thresh_full)
            boundary_full = np.fliplr(boundary_full)
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = image.rotate(angle, resample=Image.BILINEAR)
            M = cv2.getRotationMatrix2D((orig_w/2, orig_h/2), angle, 1)
            score_full  = cv2.warpAffine(score_full, M, (orig_w,orig_h), flags=cv2.INTER_NEAREST)
            thresh_full = (cv2.warpAffine((thresh_full*255).astype(np.uint8), M, (orig_w,orig_h), flags=cv2.INTER_LINEAR)/255.0)
            boundary_full = cv2.warpAffine(boundary_full, M, (orig_w,orig_h), flags=cv2.INTER_NEAREST)
        if random.random() < self.p_color:
            image = self.color_jitter(image)

        # 5) Crop logic: train vs. val
        if self.crop_range is None:
            # validation / full image
            left, top = 0, 0
            patch_w, patch_h = orig_w, orig_h
        else:
            # train: random crop size between min and max
            min_c, max_c = self.crop_range
            patch_size = random.randint(min_c, max_c)
            if patch_size < orig_w and patch_size < orig_h:
                left = random.randint(0, orig_w - patch_size)
                top  = random.randint(0, orig_h - patch_size)
                # skip too-empty patches
                if score_full[top:top+patch_size, left:left+patch_size].sum() < patch_size*patch_size*self.empty_thresh:
                    return self.__getitem__(random.randint(0, len(self.ids)-1))
                image       = image.crop((left, top, left+patch_size, top+patch_size))
                thresh_full = thresh_full[top:top+patch_size, left:left+patch_size]
                boundary_full = boundary_full[top:top+patch_size, left:left+patch_size]
                score_full  = score_full[top:top+patch_size, left:left+patch_size]
                patch_w, patch_h = patch_size, patch_size
            else:
                left, top = 0, 0
                patch_w, patch_h = orig_w, orig_h

        # 6) Adjust GT-boxes to patch
        orig_boxes = (np.array(self.boxes.get(img_id, []), dtype=np.float32)
                      * scale)
        patch_boxes = []
        for x1,y1,x2,y2 in orig_boxes:
            ix1 = max(x1, left);    iy1 = max(y1, top)
            ix2 = min(x2, left+patch_w); iy2 = min(y2, top+patch_h)
            if ix2>ix1 and iy2>iy1:
                patch_boxes.append([
                    ix1 - left, iy1 - top,
                    ix2 - left, iy2 - top
                ])
        boxes_t = torch.tensor(patch_boxes, dtype=torch.float32) if patch_boxes else torch.zeros((0,4))

        # 7) To tensors and return
        img_t     = self.normalize(image)
        score_t   = torch.from_numpy((score_full>0).astype(np.float32)).unsqueeze(0)
        thresh_t  = torch.from_numpy(thresh_full.astype(np.float32)).unsqueeze(0)
        bnd_t     = torch.from_numpy(boundary_full.astype(np.float32)).unsqueeze(0)
        raw_img   = torch.from_numpy(np.array(image)).permute(2,0,1).to(torch.uint8)

        return img_t, score_t, thresh_t, bnd_t, raw_img, boxes_t

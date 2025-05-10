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
        base_size=2048,
        crop_size=512,  # if None — disable spatial aug and crop
        shrink_ratio=0.3,
        p_flip=0.1,
        p_rotate=0.1,
        p_color=0.1,
        max_angle=7,
    ):
        # Normalize paths
        if isinstance(images_dirs, (str, Path)):
            images_dirs = [images_dirs]
        self.images_dirs = [Path(p) for p in images_dirs]
        if isinstance(ann_files, (str, Path)):
            ann_files = [ann_files]
        self.ann_files = ann_files

        # Params
        self.base_size = base_size
        self.crop_size = crop_size
        self.shrink_ratio = shrink_ratio

        # Transforms all on CPU
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Augmentations
        if crop_size is not None:
            self.augment = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=p_flip),
                K.RandomRotation(degrees=max_angle, p=p_rotate),
                K.RandomCrop(size=(crop_size, crop_size), p=1.0),
            )
            self.color_augment = K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p_color
            )
        else:
            self.augment = None
            self.color_augment = None

        # Load annotations and assign unique IDs per file to avoid overrides
        self.images = {}
        self.anns = {}
        self.img_map = {}
        self.ids = []  # list of unique IDs

        for file_idx, (ann_file, img_dir) in enumerate(zip(self.ann_files, self.images_dirs)):
            data = json.load(open(ann_file, 'r', encoding='utf8'))
            for im in data.get('images', []):
                orig_id = im['id']
                uid = f"{file_idx}_{orig_id}"  # unique across files
                self.images[uid] = im
                self.img_map[uid] = img_dir
                self.ids.append(uid)
            for ann in data.get('annotations', []):
                orig_img_id = ann['image_id']
                uid = f"{file_idx}_{orig_img_id}"
                self.anns.setdefault(uid, []).append(ann)

    def __len__(self):
        return len(self.ids)

    def _resize_if_needed(self, image: Image.Image):
        w, h = image.size
        scale = self.base_size / max(w, h)
        if scale != 1.0:
            nw, nh = int(w * scale), int(h * scale)
            image = image.resize((nw, nh), Image.BILINEAR)
            return image, scale, nw, nh
        return image, scale, w, h

    def _build_masks(self, uid, scale, H, W):
        shr = np.zeros((H, W), dtype=np.uint8)
        dil = np.zeros((H, W), dtype=np.uint8)
        for ann in self.anns.get(uid, []):
            poly = (np.array(ann['segmentation'][0], dtype=np.float32).reshape(-1, 2) * scale)
            ctr = poly.mean(axis=0)
            shr_poly = ctr + (poly - ctr) * (1 - self.shrink_ratio)
            dil_poly = ctr + (poly - ctr) * (1 + self.shrink_ratio)
            cv2.fillPoly(shr, [shr_poly.astype(np.int32)], 1)
            cv2.fillPoly(dil, [dil_poly.astype(np.int32)], 1)
        shr_t = torch.from_numpy(shr).unsqueeze(0).unsqueeze(0).float()
        dil_t = torch.from_numpy(dil).unsqueeze(0).unsqueeze(0).float()
        return shr_t, dil_t

    def __getitem__(self, idx):
        uid = self.ids[idx]
        info = self.images[uid]

        # 1) Load & resize
        img = Image.open(self.img_map[uid] / info['file_name']).convert('RGB')
        img, scale, W, H = self._resize_if_needed(img)

        # 2) Masks
        shr_t, dil_t = self._build_masks(uid, scale, H, W)

        # 3) Distance transform
        d1 = cv2.distanceTransform(shr_t[0, 0].numpy().astype(np.uint8), cv2.DIST_L2, 5)
        d2 = cv2.distanceTransform(dil_t[0, 0].numpy().astype(np.uint8), cv2.DIST_L2, 5)
        d1_t = torch.from_numpy(d1).unsqueeze(0).unsqueeze(0).float()
        d2_t = torch.from_numpy(d2).unsqueeze(0).unsqueeze(0).float()

        # 4) Targets
        score_t = shr_t
        thresh_t = d1_t / (d1_t + d2_t + 1e-6) * dil_t
        bnd_t = (F.max_pool2d(score_t, 3, 1, 1) - score_t).clamp(min=0)

        # 5) Combine & augment
        img_t = TF.to_tensor(img).unsqueeze(0)
        combined = torch.cat([img_t, score_t, thresh_t, bnd_t], dim=1)
        if self.augment is not None:
            combined = self.augment(combined)
            img_t = combined[:, :3]
            score_t = combined[:, 3:4]
            thresh_t = combined[:, 4:5]
            bnd_t = combined[:, 5:6]
            img_t = self.color_augment(img_t)
        else:
            img_t = combined[:, :3]

        # 6) Normalize
        img_out = self.normalize(TF.to_pil_image(img_t.squeeze(0)))
        return img_out, score_t.squeeze(0), thresh_t.squeeze(0), bnd_t.squeeze(0)

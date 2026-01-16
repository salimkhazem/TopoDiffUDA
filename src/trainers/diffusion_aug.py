"""Diffusion augmentation helpers."""

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils.image import load_image


class DiffusionAugmenter:
    def __init__(self, manifest_path: Path, prob: float = 1.0):
        self.manifest_path = manifest_path
        self.prob = prob
        self.mapping = self._load_manifest(manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, str]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        for k, v in data.items():
            if isinstance(v, list):
                mapping[k] = [item["generated_path"] for item in v]
            elif isinstance(v, dict):
                mapping[k] = v["generated_path"]
            else:
                mapping[k] = v
        return mapping

    def wrap(self, dataset: Dataset) -> Dataset:
        return DiffusionAugDataset(dataset, self.mapping, self.prob)


class DiffusionAugDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mapping: Dict[str, str], prob: float):
        self.base_dataset = base_dataset
        self.mapping = mapping
        self.prob = prob

    @staticmethod
    def _resize_to_target(image: "np.ndarray", target_hw) -> "np.ndarray":
        th, tw = target_hw
        h, w = image.shape[:2]
        if (h, w) == (th, tw):
            return image
        img = Image.fromarray(image)
        img = img.resize((tw, th), resample=Image.BILINEAR)
        return np.array(img)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["image"]
        meta = sample.get("meta", {})
        source_path = meta.get("source_path")
        if source_path and source_path in self.mapping and random.random() < self.prob:
            entry = self.mapping[source_path]
            if isinstance(entry, list):
                gen_path = Path(random.choice(entry))
            else:
                gen_path = Path(entry)
            if gen_path.exists():
                gen_img = load_image(gen_path)
                target_h, target_w = image.shape[-2:]
                gen_img = self._resize_to_target(gen_img, (target_h, target_w))
                gen_img = torch.from_numpy(gen_img).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
                std = torch.tensor(IMAGENET_STD)[:, None, None]
                gen_img = (gen_img - mean) / std
                sample["image"] = gen_img
        return sample

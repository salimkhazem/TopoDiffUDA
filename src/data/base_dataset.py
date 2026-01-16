"""Base dataset utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.image import load_image, load_mask, resize_pair
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.paths import get_outputs_root


class BaseSegDataset(Dataset):
    """Base segmentation dataset with optional caching."""

    def __init__(
        self,
        name: str,
        items: List[Tuple[Path, Path]],
        split: str,
        transforms=None,
        image_size: Optional[Tuple[int, int]] = None,
        ignore_index: Optional[int] = None,
        cache_preprocessed: bool = True,
    ) -> None:
        self.name = name
        self.items = items
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.cache_preprocessed = cache_preprocessed

        self.cache_dir = get_outputs_root() / "cache" / "datasets" / name / split
        ensure_dir(self.cache_dir / "images")
        ensure_dir(self.cache_dir / "masks")

    def __len__(self) -> int:
        return len(self.items)

    def _cache_paths(self, index: int) -> Tuple[Path, Path]:
        image_cache = self.cache_dir / "images" / f"{index:06d}.png"
        mask_cache = self.cache_dir / "masks" / f"{index:06d}.png"
        return image_cache, mask_cache

    def _load_item(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.items[index]
        image = load_image(image_path)
        mask = load_mask(mask_path)
        if self.image_size is not None:
            image, mask = resize_pair(image, mask, self.image_size)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.cache_preprocessed:
            image_cache, mask_cache = self._cache_paths(index)
            if image_cache.exists() and mask_cache.exists():
                image = load_image(image_cache)
                mask = load_mask(mask_cache)
            else:
                image, mask = self._load_item(index)
                image_cache.parent.mkdir(parents=True, exist_ok=True)
                mask_cache.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                Image.fromarray(image.astype(np.uint8)).save(image_cache)
                Image.fromarray(mask.astype(np.uint8)).save(mask_cache)
        else:
            image, mask = self._load_item(index)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        image_path, mask_path = self.items[index]
        return {
            "image": image,
            "mask": mask,
            "meta": {"index": index, "source_path": str(image_path), "mask_path": str(mask_path)},
        }


def cache_split_manifest(name: str, split_items: Dict[str, List[Tuple[str, str]]]) -> Path:
    cache_dir = get_outputs_root() / "cache" / "splits"
    ensure_dir(cache_dir)
    path = cache_dir / f"{name}_splits.json"
    write_json(path, split_items)
    return path


def load_split_manifest(name: str) -> Optional[Dict[str, List[List[str]]]]:
    path = get_outputs_root() / "cache" / "splits" / f"{name}_splits.json"
    if path.exists():
        return read_json(path)
    return None

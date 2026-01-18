"""SYNTHIA dataset loader (source domain for Cityscapes UDA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.image import load_image, load_mask, resize_pair
from src.utils.paths import get_data_root


def _normalize_stem(stem: str) -> str:
    for suffix in ["_RGB", "_rgb", "_leftImg8bit"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks = {_normalize_stem(p.stem): p for p in masks_dir.glob("*.*")}
    items = []
    for img in sorted(images_dir.glob("*.*")):
        key = _normalize_stem(img.stem)
        mask = masks.get(key)
        if mask is not None:
            items.append((img, mask))
    return items


def _find_root() -> Path:
    data_root = get_data_root()
    candidates = [
        data_root / "synthia",
        data_root / "SYNTHIA",
        data_root / "SYNTHIA_RAND_CITYSCAPES",
        data_root / "Synthia_Rand_Cityscapes",
        data_root / "synthia_rand_cityscapes",
    ]
    for cand in candidates:
        if (cand / "RGB").exists() or (cand / "images").exists():
            return cand
    raise FileNotFoundError(
        "SYNTHIA dataset not found. Expected data/SYNTHIA_RAND_CITYSCAPES with RGB/ and GT/LABELS/"
    )


def _find_dirs(root: Path) -> Tuple[Path, Path]:
    rgb_dir = root / "RGB"
    if not rgb_dir.exists():
        rgb_dir = root / "images"
    labels_dir = root / "GT" / "LABELS"
    if not labels_dir.exists():
        labels_dir = root / "GT" / "labels"
    if not labels_dir.exists():
        labels_dir = root / "labels"
    if not rgb_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "SYNTHIA RGB/labels not found. Expected RGB/ and GT/LABELS (or images/labels)."
        )
    return rgb_dir, labels_dir


def _load_label_map(path: Path) -> Dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "label_map" in data:
        data = data["label_map"]
    if not isinstance(data, dict):
        raise ValueError("Label map file must contain a dict of id->trainId.")
    mapping: Dict[int, int] = {}
    for k, v in data.items():
        mapping[int(k)] = int(v)
    return mapping


def _map_mask(mask: np.ndarray, mapping: Dict[int, int], ignore_index: int) -> np.ndarray:
    mapped = np.full_like(mask, ignore_index)
    for k, v in mapping.items():
        mapped[mask == k] = v
    return mapped


def get_splits(config: Dict) -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("synthia")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = _find_root()
    rgb_dir, labels_dir = _find_dirs(root)

    splits: Dict[str, List[Tuple[Path, Path]]] = {}
    if (rgb_dir / "train").exists():
        for split in ["train", "val", "test"]:
            img_split = rgb_dir / split
            mask_split = labels_dir / split
            if not img_split.exists() or not mask_split.exists():
                if split == "val":
                    img_split = rgb_dir / "valid"
                    mask_split = labels_dir / "valid"
            items = _collect_pairs(img_split, mask_split) if img_split.exists() else []
            if items:
                splits[split] = items

    if "train" not in splits:
        items = _collect_pairs(rgb_dir, labels_dir)
        if not items:
            raise FileNotFoundError("SYNTHIA images/masks not found. Check RGB and GT/LABELS.")
        val_ratio = float(config.get("dataset", {}).get("val_ratio", 0.1))
        test_ratio = float(config.get("dataset", {}).get("test_ratio", 0.0))
        seed = int(config.get("seed", 0))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(items)).tolist()
        val_size = max(1, int(len(items) * val_ratio))
        test_size = max(0, int(len(items) * test_ratio))
        train_end = max(1, len(items) - val_size - test_size)
        train_idx = perm[:train_end]
        val_idx = perm[train_end : train_end + val_size]
        test_idx = perm[train_end + val_size :]
        splits["train"] = [items[i] for i in train_idx]
        splits["val"] = [items[i] for i in val_idx]
        splits["test"] = [items[i] for i in test_idx] if test_idx else splits["val"]

    if "val" not in splits:
        splits["val"] = splits["train"][-max(1, int(0.1 * len(splits["train"]))):]
    if "test" not in splits:
        splits["test"] = splits["val"]

    cache_split_manifest("synthia", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class SynthiaDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        cfg = config or {}
        splits = get_splits(cfg)
        image_size = tuple(cfg.get("dataset", {}).get("image_size", [512, 512]))
        ignore_index = int(cfg.get("dataset", {}).get("ignore_index", 255))
        super().__init__(
            "synthia",
            splits[split],
            split,
            transforms=transforms,
            image_size=image_size,
            ignore_index=ignore_index,
        )
        self.ignore_index = ignore_index
        self.num_classes = cfg.get("model", {}).get("num_classes")
        label_map_path = cfg.get("dataset", {}).get("label_map_path")
        self.label_map = _load_label_map(Path(label_map_path)) if label_map_path else None

    def _load_item(self, index: int):
        image_path, mask_path = self.items[index]
        image = load_image(image_path)
        mask = load_mask(mask_path)
        if self.image_size is not None:
            image, mask = resize_pair(image, mask, self.image_size)
        if self.label_map:
            mask = _map_mask(mask, self.label_map, self.ignore_index)
        if self.num_classes is not None:
            invalid = (mask != self.ignore_index) & ((mask < 0) | (mask >= int(self.num_classes)))
            if np.any(invalid):
                raise ValueError(
                    "SYNTHIA mask contains labels outside [0, num_classes). "
                    "Provide a correct label_map_path for SYNTHIA -> Cityscapes."
                )
        return image, mask

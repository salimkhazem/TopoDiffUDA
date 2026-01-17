"""DeepGlobe road extraction dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks = {p.stem: p for p in masks_dir.glob("*.*")}
    items = []
    for img in sorted(images_dir.glob("*.*")):
        mask = masks.get(img.stem)
        if mask is not None:
            items.append((img, mask))
    return items


def _collect_pairs_flat(folder: Path) -> List[Tuple[Path, Path]]:
    masks = {}
    for mask in folder.glob("*_mask.*"):
        base = mask.stem.replace("_mask", "")
        masks[base] = mask
    items = []
    for img in sorted(folder.glob("*_sat.*")):
        base = img.stem.replace("_sat", "")
        mask = masks.get(base)
        if mask is not None:
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("deepglobe")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = get_data_root() / "deepglobe"
    if not root.exists():
        raise FileNotFoundError("DeepGlobe dataset not found. Place data in data/deepglobe")

    splits = {}
    for split in ["train", "val", "test"]:
        images_dir = root / "images" / split
        masks_dir = root / "masks" / split
        if not images_dir.exists():
            images_dir = root / split / "images"
            masks_dir = root / split / "masks"
        items = _collect_pairs(images_dir, masks_dir) if images_dir.exists() else []

        if not items:
            alt_split = "valid" if split == "val" else split
            flat_dir = root / alt_split
            if flat_dir.exists():
                items = _collect_pairs_flat(flat_dir)

        if items:
            splits[split] = items

    if "train" not in splits:
        raise FileNotFoundError("DeepGlobe train split not found under images/masks or train/ folders.")
    if "val" not in splits:
        splits["val"] = splits["train"][-max(1, int(0.2 * len(splits["train"]))):]
    if "test" not in splits:
        splits["test"] = splits["val"]

    cache_split_manifest("deepglobe", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class DeepGlobeRoads(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("deepglobe", splits[split], split, transforms=transforms, image_size=image_size)

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample["mask"] = (sample["mask"] > 0).long()
        return sample

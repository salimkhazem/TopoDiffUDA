"""DRIVE retinal vessel dataset."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _extract_id(name: str) -> str:
    match = re.findall(r"\d+", name)
    return match[0] if match else name


def _collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks = { _extract_id(p.stem): p for p in masks_dir.glob("*.*") }
    items = []
    for img in sorted(images_dir.glob("*.*")):
        key = _extract_id(img.stem)
        mask = masks.get(key)
        if mask is not None:
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("drive")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    data_root = get_data_root()
    root = None
    for cand in [data_root / "DRIVE", data_root / "drive"]:
        if cand.exists():
            root = cand
            break
    if root is None:
        raise FileNotFoundError("DRIVE dataset not found. Expected data/DRIVE or data/drive")

    train_images = root / "training" / "images"
    train_masks = root / "training" / "1st_manual"
    test_images = root / "test" / "images"
    test_masks = root / "test" / "1st_manual"

    train_items = _collect_pairs(train_images, train_masks)
    test_items = _collect_pairs(test_images, test_masks)

    # Deterministic split: last 20% for val
    val_size = max(1, int(0.2 * len(train_items)))
    val_items = train_items[-val_size:]
    train_items = train_items[:-val_size]

    splits = {"train": train_items, "val": val_items, "test": test_items}
    cache_split_manifest("drive", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class DRIVEDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("drive", splits[split], split, transforms=transforms, image_size=image_size)

"""STARE retinal vessel dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _find_mask_dir(root: Path) -> Path:
    for name in ["masks", "labels", "annotations"]:
        cand = root / name
        if cand.exists():
            return cand
    return root


def _collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks = {p.stem: p for p in masks_dir.glob("*.*")}
    items = []
    for img in sorted(images_dir.glob("*.*")):
        mask = masks.get(img.stem) or masks.get(img.stem.replace("im", "mask"))
        if mask is not None:
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("stare")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = get_data_root() / "STARE"
    if not root.exists():
        raise FileNotFoundError("STARE dataset not found. Place data in data/STARE")

    images_dir = root
    masks_dir = _find_mask_dir(root)
    items = _collect_pairs(images_dir, masks_dir)
    if not items:
        raise FileNotFoundError("STARE masks not found. Place mask files under data/STARE/masks or data/STARE/labels")

    val_size = max(1, int(0.2 * len(items)))
    val_items = items[-val_size:]
    train_items = items[:-val_size]
    test_items = val_items

    splits = {"train": train_items, "val": val_items, "test": test_items}
    cache_split_manifest("stare", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class STAREDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("stare", splits[split], split, transforms=transforms, image_size=image_size)

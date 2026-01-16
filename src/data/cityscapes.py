"""Cityscapes dataset loader."""

from pathlib import Path
from typing import Dict, List, Tuple

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _find_root() -> Path:
    data_root = get_data_root()
    for cand in [data_root / "cityscapes", data_root]:
        if (cand / "leftImg8bit").exists() and (cand / "gtFine").exists():
            return cand
    raise FileNotFoundError(
        "Cityscapes dataset not found. Place leftImg8bit and gtFine under data/cityscapes or data/"
    )


def _collect_pairs(split: str) -> List[Tuple[Path, Path]]:
    root = _find_root()
    images_dir = root / "leftImg8bit" / split
    masks_dir = root / "gtFine" / split
    items: List[Tuple[Path, Path]] = []
    for img in images_dir.glob("*/*_leftImg8bit.png"):
        city = img.parent.name
        stem = img.stem.replace("_leftImg8bit", "")
        mask = masks_dir / city / f"{stem}_gtFine_labelTrainIds.png"
        if not mask.exists():
            mask = masks_dir / city / f"{stem}_gtFine_labelIds.png"
        if mask.exists():
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("cityscapes")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    splits = {"train": _collect_pairs("train"), "val": _collect_pairs("val"), "test": _collect_pairs("val")}
    if not splits["train"]:
        raise FileNotFoundError("Cityscapes images not found. Ensure leftImg8bit/gtFine are present.")

    cache_split_manifest("cityscapes", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class CityscapesDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("cityscapes", splits[split], split, transforms=transforms, image_size=image_size, ignore_index=255)

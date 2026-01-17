"""SpaceNet roads dataset (expects preprocessed masks)."""

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


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("spacenet")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    data_root = get_data_root()
    root = None
    for cand in [data_root / "spacenet", data_root / "SN3_roads"]:
        if cand.exists():
            root = cand
            break
    if root is None:
        raise FileNotFoundError("SpaceNet dataset not found. Place data in data/spacenet or data/SN3_roads")

    splits = {}
    for split in ["train", "val", "test"]:
        images_dir = root / "images" / split
        masks_dir = root / "masks" / split
        if not images_dir.exists():
            images_dir = root / "processed" / "images" / split
            masks_dir = root / "processed" / "masks" / split
        items = _collect_pairs(images_dir, masks_dir)
        if not items:
            raise FileNotFoundError(
                "SpaceNet masks not found. Preprocess geojson to masks and place under data/spacenet/processed/masks/<split>"
            )
        splits[split] = items

    cache_split_manifest("spacenet", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class SpaceNetRoads(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("spacenet", splits[split], split, transforms=transforms, image_size=image_size)

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample["mask"] = (sample["mask"] > 0).long()
        return sample

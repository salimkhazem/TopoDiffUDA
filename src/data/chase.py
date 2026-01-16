"""CHASEDB1 retinal vessel dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    images = list(root.glob("*image*.*"))
    masks = list(root.glob("*mask*.*")) + list(root.glob("*label*.*"))
    if not images or not masks:
        return []
    mask_map = {m.stem.replace("_1stHO", "").replace("_manual", ""): m for m in masks}
    items = []
    for img in sorted(images):
        key = img.stem.replace("_image", "")
        mask = mask_map.get(key)
        if mask is not None:
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("chase")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = None
    for cand in [get_data_root() / "CHASE", get_data_root() / "CHASEDB1", get_data_root() / "chase"]:
        if cand.exists():
            root = cand
            break
    if root is None:
        raise FileNotFoundError("CHASE dataset not found. Place data in data/CHASEDB1")

    items = _collect_pairs(root)
    if not items:
        raise FileNotFoundError("CHASE masks not found. Ensure mask files are in the dataset folder.")

    val_size = max(1, int(0.2 * len(items)))
    val_items = items[-val_size:]
    train_items = items[:-val_size]
    test_items = val_items

    splits = {"train": train_items, "val": val_items, "test": test_items}
    cache_split_manifest("chase", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class CHASEDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("chase", splits[split], split, transforms=transforms, image_size=image_size)

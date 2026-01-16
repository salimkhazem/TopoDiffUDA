"""Optional HuggingFace dataset wrappers."""

from typing import Any, Dict, List, Tuple


def load_hf_segmentation_dataset(dataset_name: str, split: str):
    """Load a HF dataset if available; returns None if not installed or missing."""
    try:
        from datasets import load_dataset
    except Exception:
        return None

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception:
        return None

    return dataset


def hf_to_items(dataset, image_key: str, mask_key: str) -> List[Tuple[Any, Any]]:
    items = []
    for sample in dataset:
        items.append((sample[image_key], sample[mask_key]))
    return items

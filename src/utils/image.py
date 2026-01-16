"""Image IO utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: Path, rgb: bool = True) -> np.ndarray:
    img = Image.open(path)
    if rgb:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.array(mask)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.fromarray(image)
    m = Image.fromarray(mask)
    img = img.resize(size, resample=Image.BILINEAR)
    m = m.resize(size, resample=Image.NEAREST)
    return np.array(img), np.array(m)

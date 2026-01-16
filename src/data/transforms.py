"""Augmentation pipelines."""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        img = Image.fromarray(image)
        m = Image.fromarray(mask)
        img = img.resize(self.size, resample=Image.BILINEAR)
        m = m.resize(self.size, resample=Image.NEAREST)
        return np.array(img), np.array(m)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]
        th, tw = self.size
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = image.shape[:2]
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        image = image[i : i + th, j : j + tw]
        mask = mask[i : i + th, j : j + tw]
        return image, mask


class ApplyColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        img = Image.fromarray(image)
        img = self.jitter(img)
        return np.array(img), mask


class GaussianBlur:
    def __init__(self, radius: float = 1.0, p: float = 0.2):
        self.radius = radius
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            img = Image.fromarray(image)
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
            image = np.array(img)
        return image, mask


class ToTensorNormalize:
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean)[:, None, None]
            std = torch.tensor(self.std)[:, None, None]
            image_t = (image_t - mean) / std
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: Tuple[int, int],
    split: str,
    strong: bool = False,
    disable_color: bool = False,
) -> Compose:
    transforms: List[Callable] = []
    if split == "train":
        transforms.extend([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.2), RandomRotate90(0.3)])
        if strong:
            transforms.append(RandomCrop(image_size))
            if not disable_color:
                transforms.append(ApplyColorJitter())
            transforms.append(GaussianBlur())
        else:
            transforms.append(Resize(image_size))
    else:
        transforms.append(Resize(image_size))

    transforms.append(ToTensorNormalize(IMAGENET_MEAN, IMAGENET_STD))
    return Compose(transforms)

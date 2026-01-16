"""Dataset builders and dataloader helpers."""

from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from src.data.transforms import build_transforms
from src.utils.seed import seed_worker, get_generator

from .chase import CHASEDataset
from .deepglobe_roads import DeepGlobeRoads
from .drive import DRIVEDataset
from .hf_retina import HFRetinaDataset
from .gta5 import GTA5Dataset
from .cityscapes import CityscapesDataset
from .spacenet_roads import SpaceNetRoads
from .ssdd import SSDDDataset
from .stare import STAREDataset


DATASET_REGISTRY = {
    "drive": DRIVEDataset,
    "stare": STAREDataset,
    "chase": CHASEDataset,
    "deepglobe": DeepGlobeRoads,
    "spacenet": SpaceNetRoads,
    "gta5": GTA5Dataset,
    "cityscapes": CityscapesDataset,
    "gta5_cityscapes": GTA5Dataset,
    "ssdd": SSDDDataset,
}

def _select_dataset_cls(name: str, cfg: Dict):
    if cfg.get("dataset", {}).get("hf_name") and name in {"drive", "stare", "chase"}:
        return HFRetinaDataset
    return DATASET_REGISTRY[name]


def build_dataset(name: str, split: str, cfg: Dict, strong: bool = False):
    if name == "gta5_cityscapes" and "target" in cfg.get("dataset", {}):
        name = cfg["dataset"]["target"]
    dataset_cls = _select_dataset_cls(name, cfg)
    image_size = tuple(cfg["dataset"]["image_size"])
    disable_color = cfg["dataset"].get("disable_color_aug", False)
    transforms = build_transforms(image_size, split, strong=strong, disable_color=disable_color)
    if dataset_cls is HFRetinaDataset:
        return dataset_cls(name=name, split=split, transforms=transforms, config=cfg)
    return dataset_cls(split=split, transforms=transforms, config=cfg)


class DataModule:
    def __init__(self, cfg: Dict, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset_name = self.cfg["dataset"]["name"]
        train_ds = build_dataset(dataset_name, "train", self.cfg, strong=False)
        val_ds = build_dataset(dataset_name, "val", self.cfg, strong=False)
        test_ds = build_dataset(dataset_name, "test", self.cfg, strong=False)

        batch_size = self.cfg["train"]["batch_size"]
        num_workers = self.cfg["train"].get("num_workers", 4)
        generator = get_generator(self.seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    def get_uda_loaders(self) -> Dict[str, DataLoader]:
        source_name = self.cfg["dataset"].get("source", self.cfg["dataset"]["name"])
        target_name = self.cfg["dataset"].get("target", self.cfg["dataset"]["name"])
        image_size = tuple(self.cfg["dataset"]["image_size"])
        disable_color = self.cfg["dataset"].get("disable_color_aug", False)
        bs = self.cfg["train"]["batch_size"]
        num_workers = self.cfg["train"].get("num_workers", 4)
        generator = get_generator(self.seed)

        source_cls = _select_dataset_cls(source_name, self.cfg)
        target_cls = _select_dataset_cls(target_name, self.cfg)

        if source_cls is HFRetinaDataset:
            source_train = source_cls(
                name=source_name,
                split="train",
                transforms=build_transforms(image_size, "train", strong=False, disable_color=disable_color),
                config=self.cfg,
            )
            source_train_strong = source_cls(
                name=source_name,
                split="train",
                transforms=build_transforms(image_size, "train", strong=True, disable_color=disable_color),
                config=self.cfg,
            )
        else:
            source_train = source_cls(
                split="train",
                transforms=build_transforms(image_size, "train", strong=False, disable_color=disable_color),
                config=self.cfg,
            )
            source_train_strong = source_cls(
                split="train",
                transforms=build_transforms(image_size, "train", strong=True, disable_color=disable_color),
                config=self.cfg,
            )

        if target_cls is HFRetinaDataset:
            target_train = target_cls(
                name=target_name,
                split="train",
                transforms=build_transforms(image_size, "train", strong=False, disable_color=disable_color),
                config=self.cfg,
            )
            target_train_strong = target_cls(
                name=target_name,
                split="train",
                transforms=build_transforms(image_size, "train", strong=True, disable_color=disable_color),
                config=self.cfg,
            )
            target_val = target_cls(
                name=target_name,
                split="val",
                transforms=build_transforms(image_size, "val", strong=False, disable_color=disable_color),
                config=self.cfg,
            )
        else:
            target_train = target_cls(
                split="train",
                transforms=build_transforms(image_size, "train", strong=False, disable_color=disable_color),
                config=self.cfg,
            )
            target_train_strong = target_cls(
                split="train",
                transforms=build_transforms(image_size, "train", strong=True, disable_color=disable_color),
                config=self.cfg,
            )
            target_val = target_cls(
                split="val",
                transforms=build_transforms(image_size, "val", strong=False, disable_color=disable_color),
                config=self.cfg,
            )

        loaders = {
            "source": DataLoader(
                source_train,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator,
            ),
            "source_strong": DataLoader(
                source_train_strong,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator,
            ),
            "target": DataLoader(
                target_train,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator,
            ),
            "target_strong": DataLoader(
                target_train_strong,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator,
            ),
            "target_val": DataLoader(
                target_val,
                batch_size=bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }
        return loaders

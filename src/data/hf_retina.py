"""HuggingFace retinal vessel datasets (DRIVE, STARE, CHASE_DB1)."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import ClassLabel, Image as HFImage, load_dataset
except Exception:  # pragma: no cover - optional dependency
    ClassLabel = None
    HFImage = None
    load_dataset = None

from src.utils.io import ensure_dir, read_json, write_json
from src.utils.paths import get_outputs_root


HF_RETINA_MAP = {
    "drive": "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction",
    "stare": "Zomba/STARE-structured-analysis-of-the-retina",
    "chase": "Zomba/CHASE_DB1-retinal-dataset",
}


def _hf_cache_dir(cfg: Dict[str, Any]) -> Optional[str]:
    cfg_dir = cfg.get("dataset", {}).get("hf_cache_dir")
    if cfg_dir:
        return cfg_dir
    return os.environ.get("TOPODIFFUDA_HF_CACHE") or os.environ.get("HF_DATASETS_CACHE")


def _finalize_indices(n: int, seed: int, val_ratio: float, name: str) -> Dict[str, List[int]]:
    cache_dir = get_outputs_root() / "cache" / "splits"
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"{name}_hf_seed{seed}.json"
    if cache_path.exists():
        cached = read_json(cache_path)
        if "train" in cached and "val" in cached:
            return cached
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()
    val_size = max(1, int(n * val_ratio))
    splits = {"train": perm[:-val_size], "val": perm[-val_size:]}
    write_json(cache_path, splits)
    return splits


class HFRetinaDataset(Dataset):
    """HF wrapper with deterministic train/val split."""

    def __init__(
        self,
        name: str,
        split: str,
        transforms: Optional[Callable[[Any, Any], Tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if load_dataset is None:
            raise ImportError("datasets is required for HuggingFace datasets")
        cfg = config or {}
        ds_cfg = cfg.get("dataset", {})
        hf_name = HF_RETINA_MAP.get(name, ds_cfg.get("hf_name", name))
        train_split = ds_cfg.get("hf_train_split", "train")
        test_split = ds_cfg.get("hf_test_split", "validation")
        val_ratio = float(ds_cfg.get("val_ratio", 0.2))
        seed = int(cfg.get("seed", 0))
        cache_dir = _hf_cache_dir(cfg)
        self.transforms = transforms
        self.image_key = ds_cfg.get("image_key", "image")
        self.mask_key = ds_cfg.get("mask_key", "label")

        try:
            if split in {"train", "val"}:
                ds = load_dataset(hf_name, split=train_split, cache_dir=cache_dir, download_mode="reuse_cache_if_exists")
            else:
                ds = load_dataset(hf_name, split=test_split, cache_dir=cache_dir, download_mode="reuse_cache_if_exists")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HF dataset {hf_name}. "
                "Ensure it is cached locally or set TOPODIFFUDA_HF_CACHE to the cache dir."
            ) from exc

        self.ds = ds
        self._paired = True
        self._pairs: List[Tuple[int, int]] = []
        self._init_pairing()

        if split in {"train", "val"}:
            if self._paired:
                splits = _finalize_indices(len(self.ds), seed, val_ratio, name)
                self.ds = self.ds.select(splits[split])
            else:
                key = f"{name}_pairs"
                splits = _finalize_indices(len(self._pairs), seed, val_ratio, key)
                self._pairs = [self._pairs[i] for i in splits[split]]

    def _init_pairing(self) -> None:
        if self.mask_key in self.ds.features and HFImage is not None:
            feat = self.ds.features[self.mask_key]
            if isinstance(feat, HFImage):
                self._paired = True
                return
        if "label" in self.ds.features and ClassLabel is not None:
            if isinstance(self.ds.features["label"], ClassLabel):
                try:
                    names = list(getattr(self.ds.features["label"], "names", []))
                except Exception:
                    names = []
                if names == ["input", "label"]:
                    ds_paths = self.ds.cast_column(self.image_key, HFImage(decode=False))
                    inputs: Dict[str, int] = {}
                    labels: Dict[str, int] = {}
                    for i in range(len(ds_paths)):
                        ex = ds_paths[i]
                        kind = int(ex["label"])
                        path = ex[self.image_key]["path"]
                        stem = Path(path).stem
                        if kind == 0:
                            inputs[stem] = i
                        else:
                            labels[stem] = i
                    common = sorted(set(inputs) & set(labels))
                    if common:
                        self._pairs = [(inputs[k], labels[k]) for k in common]
                        self._paired = False
                        return
                    input_idx = [i for i in range(len(self.ds)) if int(self.ds[i]["label"]) == 0]
                    label_idx = [i for i in range(len(self.ds)) if int(self.ds[i]["label"]) == 1]
                    if len(input_idx) == len(label_idx) and len(input_idx) > 0:
                        self._pairs = list(zip(input_idx, label_idx))
                        self._paired = False
                        return
        self._paired = True

    def __len__(self) -> int:
        return len(self._pairs) if not self._paired else len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._paired:
            sample = self.ds[idx]
            img = sample[self.image_key]
            mask = sample[self.mask_key]
        else:
            img_idx, mask_idx = self._pairs[idx]
            img = self.ds[img_idx][self.image_key]
            mask = self.ds[mask_idx][self.image_key]

        if hasattr(img, "convert"):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = np.array(img)
        if hasattr(mask, "convert"):
            mask_np = np.array(mask.convert("L"))
        else:
            mask_np = np.array(mask)

        mask_np = (mask_np > 0).astype(np.uint8)

        if self.transforms is not None:
            img_t, mask_t = self.transforms(img_np, mask_np)
        else:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_np).long()
        return {
            "image": img_t,
            "mask": mask_t,
            "meta": {"index": idx, "source_path": f"hf://{idx}", "mask_path": f"hf://{idx}"},
        }

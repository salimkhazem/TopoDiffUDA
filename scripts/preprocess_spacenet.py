"""Preprocess SpaceNet SN3_roads into image/mask pairs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
except Exception as exc:
    raise ImportError("rasterio is required for SpaceNet preprocessing") from exc

try:
    from scipy.ndimage import binary_dilation
except Exception as exc:
    raise ImportError("scipy is required for SpaceNet preprocessing") from exc

from src.utils.paths import get_data_root

def extract_id(name: str) -> str:
    match = re.search(r"_img(\d+)", name)
    return match.group(1) if match else Path(name).stem


def build_geojson_map(geo_dir: Path) -> Dict[str, Path]:
    mapping = {}
    for path in geo_dir.glob("*.geojson"):
        mapping[extract_id(path.name)] = path
    return mapping


def load_geojson(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rasterize_roads(data: Dict, out_shape: Tuple[int, int], transform, dst_crs) -> np.ndarray:
    shapes = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom is None:
            continue
        try:
            geom = transform_geom("EPSG:4326", dst_crs, geom, precision=6)
        except Exception:
            pass
        shapes.append(geom)

    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )
    return mask


def split_indices(n: int, seed: int, val_ratio: float, test_ratio: float) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_size = max(1, int(n * val_ratio))
    test_size = max(1, int(n * test_ratio))
    train_end = max(1, n - val_size - test_size)
    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end : train_end + val_size].tolist()
    test_idx = indices[train_end + val_size :].tolist() if test_size > 0 else val_idx
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sn3_root", default="data/SN3_roads")
    parser.add_argument("--output_root", default="data/spacenet/processed")
    parser.add_argument("--image_type", default="PS-RGB")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--road_width", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sn3_root = Path(args.sn3_root)
    output_root = Path(args.output_root)
    if not sn3_root.is_absolute() and not sn3_root.exists():
        sn3_root = get_data_root() / "SN3_roads"
    if not output_root.is_absolute() and not output_root.exists():
        output_root = get_data_root() / "spacenet" / "processed"
    images_out = output_root / "images"
    masks_out = output_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    samples: List[Tuple[Path, Path, str]] = []
    train_root = sn3_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"SN3_roads train folder not found at {train_root}")

    for aoi_dir in sorted(train_root.glob("AOI_*")):
        img_dir = aoi_dir / args.image_type
        geo_dir = aoi_dir / "geojson_roads"
        if not img_dir.exists() or not geo_dir.exists():
            continue
        geo_map = build_geojson_map(geo_dir)
        for img_path in sorted(img_dir.glob("*.tif")):
            img_id = extract_id(img_path.name)
            geo_path = geo_map.get(img_id)
            if geo_path is None:
                continue
            samples.append((img_path, geo_path, aoi_dir.name))
            if args.limit and len(samples) >= args.limit:
                break
        if args.limit and len(samples) >= args.limit:
            break

    if not samples:
        raise FileNotFoundError("No SpaceNet samples found. Check SN3_roads structure.")

    splits = split_indices(len(samples), args.seed, args.val_ratio, args.test_ratio)

    for split, indices in splits.items():
        split_img_dir = images_out / split
        split_mask_dir = masks_out / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(indices, desc=f"Processing {split}"):
            img_path, geo_path, aoi = samples[idx]
            out_stem = f"{aoi}_{img_path.stem}"
            out_img = split_img_dir / f"{out_stem}.png"
            out_mask = split_mask_dir / f"{out_stem}.png"
            if out_img.exists() and out_mask.exists() and not args.overwrite:
                continue

            with rasterio.open(img_path) as src:
                img = src.read()
                transform = src.transform
                dst_crs = src.crs.to_string() if src.crs is not None else "EPSG:4326"
                height, width = src.height, src.width

            # Convert to HWC
            img = np.transpose(img[:3], (1, 2, 0))
            img = np.clip(img, 0, 255).astype(np.uint8)

            geojson = load_geojson(geo_path)
            mask = rasterize_roads(geojson, (height, width), transform, dst_crs)

            if args.road_width > 1:
                mask = binary_dilation(mask.astype(bool), iterations=max(1, args.road_width // 2)).astype(np.uint8)

            Image.fromarray(img).save(out_img)
            Image.fromarray(mask * 255).save(out_mask)

    split_path = output_root / "splits.json"
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Saved processed SpaceNet to {output_root}")


if __name__ == "__main__":
    main()

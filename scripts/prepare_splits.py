"""Prepare dataset splits and preprocessing cache."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from pathlib import Path

from tqdm import tqdm

from src.data.datamodule import build_dataset
from src.utils.config import load_yaml, merge_dicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    args = parser.parse_args()

    config = {"dataset": {"image_size": args.image_size, "name": args.dataset}}
    ds_cfg_path = Path("configs/datasets") / f"{args.dataset}.yaml"
    if ds_cfg_path.exists():
        config = merge_dicts(config, load_yaml(ds_cfg_path))
    config["dataset"]["image_size"] = args.image_size

    for split in ["train", "val", "test"]:
        ds = build_dataset(args.dataset, split, config, strong=False)
        for _ in tqdm(ds, desc=f"Caching {split}"):
            pass


if __name__ == "__main__":
    main()

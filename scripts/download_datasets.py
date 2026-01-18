"""Dataset download helper (manual instructions for gated sources)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import get_data_root


def main() -> None:
    data_root = get_data_root()
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "drive": "Download from https://drive.grand-challenge.org/ and place under data/DRIVE",
        "stare": "Download STARE images/masks and place under data/STARE with masks in data/STARE/masks",
        "chase": "Download CHASEDB1 and place under data/CHASEDB1",
        "deepglobe": "Kaggle DeepGlobe Road Extraction dataset under data/deepglobe",
        "spacenet": "SpaceNet Roads requires preprocessing; run scripts/preprocess_spacenet.py to create data/spacenet/processed",
        "gta5": "GTA5 dataset under data/gta5/images and data/gta5/labels",
        "synthia": "SYNTHIA-RAND-CITYSCAPES under data/SYNTHIA_RAND_CITYSCAPES with RGB/ and GT/LABELS/",
        "cityscapes": "Cityscapes requires registration; place leftImg8bit/gtFine under data/cityscapes",
        "ssdd": "SSDD dataset under data/ssdd/images and data/ssdd/masks",
    }

    hf_cache = (
        os.environ.get("TOPODIFFUDA_HF_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
        or os.path.expanduser("~/.cache/huggingface/datasets")
    )

    print(f"Data root: {data_root}")
    print(f"HF cache: {hf_cache}")
    print("HF datasets (DRIVE/STARE/CHASE) are supported if cached; set TOPODIFFUDA_HF_CACHE if needed.")
    for name, instructions in datasets.items():
        present = False
        for cand in [data_root / name, data_root / name.upper(), data_root / name.capitalize()]:
            if cand.exists():
                present = True
                break
        if name == "synthia" and (data_root / "SYNTHIA_RAND_CITYSCAPES").exists():
            present = True
        if name == "cityscapes" and (data_root / "leftImg8bit").exists():
            present = True
        status = "OK" if present else "MISSING"
        print(f"[{status}] {name}: {instructions}")


if __name__ == "__main__":
    main()

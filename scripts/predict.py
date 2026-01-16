"""Run inference on a folder of images."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.model_factory import build_model
from src.utils.config import load_yaml, merge_dicts


def load_config(args) -> dict:
    base = load_yaml(Path("configs/default.yaml"))
    dataset_cfg = load_yaml(Path(f"configs/datasets/{args.dataset}.yaml"))
    model_cfg = load_yaml(Path(f"configs/models/{args.model}.yaml"))
    cfg = merge_dicts(base, dataset_cfg)
    cfg = merge_dicts(cfg, model_cfg)
    cfg["dataset"]["name"] = args.dataset
    cfg["model"]["name"] = args.model
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(input_dir.glob("*.*"))):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            if logits.shape[-2:] != img_t.shape[-2:]:
                logits = F.interpolate(logits, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            else:
                pred = torch.argmax(logits, dim=1)
        pred_np = pred.cpu().numpy()[0].astype(np.uint8) * 255
        Image.fromarray(pred_np).save(output_dir / f"{img_path.stem}_pred.png")


if __name__ == "__main__":
    main()

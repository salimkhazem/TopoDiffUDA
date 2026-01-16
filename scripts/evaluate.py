"""Evaluation script."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.data.datamodule import build_dataset
from src.models.model_factory import build_model
from src.utils.config import load_yaml, merge_dicts, save_yaml
from src.utils.io import ensure_dir, write_env_snapshot
from src.utils.metrics import compute_segmentation_metrics
from src.utils.paths import get_outputs_root, get_repo_root
from src.utils.seed import seed_everything


def load_config(args) -> dict:
    base = load_yaml(Path("configs/default.yaml"))
    dataset_cfg = load_yaml(Path(f"configs/datasets/{args.dataset}.yaml"))
    model_cfg = load_yaml(Path(f"configs/models/{args.model}.yaml"))
    exp_cfg = load_yaml(Path(args.config))
    cfg = merge_dicts(base, dataset_cfg)
    cfg = merge_dicts(cfg, model_cfg)
    cfg = merge_dicts(cfg, exp_cfg)
    cfg["dataset"]["name"] = args.dataset
    cfg["model"]["name"] = args.model
    cfg["seed"] = args.seed
    return cfg


def sliding_window_inference(image: torch.Tensor, model, window: int, stride: int) -> torch.Tensor:
    _, _, h, w = image.shape
    logits_accum = None
    count = torch.zeros((1, 1, h, w), device=image.device)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y + window, h)
            x1 = min(x + window, w)
            y0 = max(0, y1 - window)
            x0 = max(0, x1 - window)
            crop = image[:, :, y0:y1, x0:x1]
            logits = model(crop)
            if logits_accum is None:
                logits_accum = torch.zeros((logits.shape[1], h, w), device=image.device)
            logits_accum[:, y0:y1, x0:x1] += logits[0]
            count[:, :, y0:y1, x0:x1] += 1
    logits_accum = logits_accum / count
    return logits_accum.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args)
    seed_everything(cfg["seed"], deterministic=cfg["train"].get("deterministic", True))

    if args.exp_name:
        cfg.setdefault("experiment", {})["name"] = args.exp_name
    exp_name = cfg.get("experiment", {}).get("name", Path(args.config).stem)
    output_dir = get_outputs_root() / "runs" / exp_name / args.dataset / args.model / str(args.seed)
    ensure_dir(output_dir)
    save_yaml(output_dir / "config.yaml", cfg)
    write_env_snapshot(output_dir, get_repo_root())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt_path = output_dir / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = output_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = build_dataset(args.dataset, "test", cfg, strong=False)
    predictions_dir = output_dir / "predictions"
    ensure_dir(predictions_dir)

    metrics_list = []
    n_bins = cfg.get("eval", {}).get("ece_bins", 10)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)

    pr_thresholds = np.linspace(0.0, 1.0, 51)
    pr_tp = np.zeros_like(pr_thresholds)
    pr_fp = np.zeros_like(pr_thresholds)
    pr_fn = np.zeros_like(pr_thresholds)
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].cpu().numpy()
            if args.window_size is not None:
                stride = args.stride or args.window_size
                logits = sliding_window_inference(image, model, args.window_size, stride)
            else:
                logits = model(image)
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long().squeeze(1)
                probs_np = torch.cat([1 - probs, probs], dim=1).permute(0, 2, 3, 1).cpu().numpy()
                num_classes = 2
            else:
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                probs_np = probs.permute(0, 2, 3, 1).cpu().numpy()
                num_classes = probs.shape[1]

            pred_np = pred.cpu().numpy()[0].astype(np.uint8)
            pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
            pred_img.save(predictions_dir / f"{idx:06d}.png")

            metrics = compute_segmentation_metrics(pred_np, mask, num_classes=num_classes, ignore_index=cfg["dataset"].get("ignore_index"), probs=probs_np[0])
            metrics["index"] = idx
            metrics_list.append(metrics)

            # Reliability stats
            conf = probs_np[0].max(axis=-1).reshape(-1)
            mask_flat = mask.reshape(-1)
            valid = mask_flat != cfg["dataset"].get("ignore_index", 255)
            acc = (pred_np.reshape(-1) == mask_flat).astype(np.float32)
            conf = conf[valid]
            acc = acc[valid]
            for b in range(n_bins):
                mask_b = (conf > bins[b]) & (conf <= bins[b + 1])
                if mask_b.sum() == 0:
                    continue
                bin_counts[b] += mask_b.mean()
                bin_conf[b] += conf[mask_b].mean()
                bin_acc[b] += acc[mask_b].mean()

            # PR curve for binary segmentation
            if num_classes == 2:
                pos_prob = probs_np[0][..., 1]
                valid_mask = mask != cfg["dataset"].get("ignore_index", 255)
                gt = ((mask == 1) & valid_mask).astype(np.uint8)
                for t_idx, thr in enumerate(pr_thresholds):
                    pred_bin = ((pos_prob >= thr) & valid_mask).astype(np.uint8)
                    pr_tp[t_idx] += float((pred_bin * gt).sum())
                    pr_fp[t_idx] += float((pred_bin * (1 - gt)).sum())
                    pr_fn[t_idx] += float(((1 - pred_bin) * gt).sum())

    aggregate = {}
    if metrics_list:
        keys = [k for k in metrics_list[0].keys() if k != "index"]
        for key in keys:
            aggregate[key] = float(sum(m[key] for m in metrics_list) / len(metrics_list))

    reliability = {
        "bins": bins.tolist(),
        "confidence": (bin_conf / (bin_counts + 1e-8)).tolist(),
        "accuracy": (bin_acc / (bin_counts + 1e-8)).tolist(),
    }
    precision = (pr_tp / (pr_tp + pr_fp + 1e-8)).tolist()
    recall = (pr_tp / (pr_tp + pr_fn + 1e-8)).tolist()

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "per_image": metrics_list,
                "aggregate": aggregate,
                "reliability": reliability,
                "pr_curve": {"thresholds": pr_thresholds.tolist(), "precision": precision, "recall": recall},
            },
            f,
            indent=2,
        )

    summary_path = output_dir / "summary.csv"
    if aggregate:
        import csv

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate.keys()))
            writer.writeheader()
            writer.writerow(aggregate)


if __name__ == "__main__":
    main()

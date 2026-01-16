"""Generate diffusion-based target-style augmentation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.data.datamodule import DATASET_REGISTRY
from src.data.hf_retina import HFRetinaDataset
from src.utils.config import load_yaml, merge_dicts
from src.utils.paths import get_outputs_root


def load_pipeline(model_id: str, device: str):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except Exception as exc:
        print("diffusers not installed. Install it via pip install diffusers")
        return None

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    except Exception:
        print("Failed to load diffusion weights. Download the model or pass a local path.")
        return None

    pipe = pipe.to(device)
    pipe.safety_checker = None
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--style_name", required=True)
    parser.add_argument("--style_prompt", default="")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = {"dataset": {"image_size": [512, 512], "name": args.dataset}}
    ds_cfg_path = Path("configs/datasets") / f"{args.dataset}.yaml"
    if ds_cfg_path.exists():
        config = merge_dicts(config, load_yaml(ds_cfg_path))
    dataset_cls = DATASET_REGISTRY.get(args.dataset)
    if config.get("dataset", {}).get("hf_name") and args.dataset in {"drive", "stare", "chase"}:
        dataset_cls = HFRetinaDataset
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset {args.dataset}")
    if dataset_cls is HFRetinaDataset:
        dataset = dataset_cls(name=args.dataset, split="train", transforms=None, config=config)
    else:
        dataset = dataset_cls(split="train", transforms=None, config=config)

    outputs_root = get_outputs_root() / "diffusion_aug" / args.dataset / args.style_name
    images_dir = outputs_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_root / "manifest.json"

    pipe = load_pipeline(args.model_id, args.device)
    if pipe is None:
        print("Diffusion pipeline unavailable. Please install diffusers and download weights.")
        return

    manifest = {}
    global_idx = 0
    for idx in tqdm(range(len(dataset)), desc="Generating"):
        sample = dataset[idx]
        img_t = sample["image"]
        if isinstance(img_t, torch.Tensor):
            img_np = img_t.detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255.0).clip(0, 255).astype("uint8")
            img = Image.fromarray(img_np)
        else:
            img = Image.fromarray(img_t)
        meta = sample.get("meta", {})
        src_key = meta.get("source_path", f"index_{idx}")
        for sample_idx in range(args.num_samples):
            seed = args.seed + global_idx
            generator = torch.Generator(device=args.device).manual_seed(seed)
            out = pipe(
                prompt=args.style_prompt,
                image=img,
                strength=args.strength,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=generator,
            )
            gen = out.images[0]
            src_stem = Path(str(src_key)).stem
            gen_path = images_dir / f"{src_stem}_gen{sample_idx:02d}.png"
            gen.save(gen_path)
            manifest.setdefault(str(src_key), [])
            manifest[str(src_key)].append(
                {
                    "generated_path": str(gen_path),
                    "seed": seed,
                    "strength": args.strength,
                    "prompt": args.style_prompt,
                }
            )
            global_idx += 1

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

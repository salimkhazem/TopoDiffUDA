# TopoDiffUDA - Diffusion-Augmented Topology-Preserving UDA

This is a **fully reproducible** PyTorch research codebase for:
**“Diffusion-Augmented Topology-Preserving Unsupervised Domain Adaptation for Segmentation.”**

TopoDiffUDA implements diffusion-augmented, topology-preserving unsupervised domain adaptation for thin-structure segmentation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional conda:

```bash
conda env create -f environment.yml
conda activate topo-diffuda
```

## Data placement

Set `TOPODIFFUDA_DATA` to your data root or place datasets under `../data` relative to this repo.
If you use HuggingFace retina datasets, set `TOPODIFFUDA_HF_CACHE` to the cache directory (e.g., `/mnt/storage_2_10T/skhazem/Projects/icpr_5/cache/hf`).

Expected structures:

- DRIVE: `data/DRIVE/training/images`, `data/DRIVE/training/1st_manual`, `data/DRIVE/test/images`, `data/DRIVE/test/1st_manual`
- STARE: `data/STARE` images + `data/STARE/masks` or `data/STARE/labels`
- CHASEDB1: `data/CHASEDB1` with image/mask files in one folder
- DeepGlobe: `data/deepglobe/images/<split>` and `data/deepglobe/masks/<split>`
- SpaceNet: preprocessed masks under `data/spacenet/processed/images/<split>` and `data/spacenet/processed/masks/<split>`
- GTA5: `data/gta5/images` and `data/gta5/labels`
- Cityscapes: `data/cityscapes/leftImg8bit` and `data/cityscapes/gtFine`
- SSDD: `data/ssdd/images/<split>` and `data/ssdd/masks/<split>`

Check dataset availability:

```bash
python scripts/download_datasets.py
```

Prepare cached splits/preprocessing:

```bash
python scripts/prepare_splits.py --dataset drive --image_size 512 512
```

## Diffusion augmentation (offline)

```bash
python scripts/generate_diffusion_aug.py --dataset drive --style_name default \
  --style_prompt "retinal imaging, high contrast" --strength 0.4 --num_samples 1 --seed 0
```

Generated images and manifest are stored under:

```
outputs/diffusion_aug/<dataset>/<style_name>/
```

## Training (source-only)

```bash
python scripts/train.py --config configs/experiments/main_full_method.yaml \
  --dataset drive --model unet --seed 0 --log_tb
```

Example command template (exact):

python scripts/train.py --config configs/experiments/main_full_method.yaml \
 --dataset gta5_cityscapes --model deeplabv3p --seed 0 --log_tb

## UDA adaptation

```bash
python scripts/adapt.py --config configs/experiments/main_full_method.yaml \
  --dataset gta5_cityscapes --model deeplabv3p --seed 0 --log_tb
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/experiments/main_full_method.yaml \
  --dataset drive --model unet --seed 0
```

## Figures and tables

```bash
python scripts/make_figures.py --run_dir outputs/runs/main_full_method/drive/unet/0
python scripts/aggregate_results.py --root outputs/runs/main_full_method
python scripts/make_tables.py --root outputs/runs
```

## Benchmarks and ablations

```bash
bash scripts/run_all_benchmarks.sh
bash scripts/run_all_ablations.sh
```

## Outputs

All artifacts are stored under `outputs/`:

```
outputs/
  runs/source_only/<dataset>/<model>/<seed>/
  runs/main_full_method/<dataset>/<model>/<seed>/
    config.yaml
    env.json
    metrics.json
    summary.csv
    best.ckpt
    predictions/
    figures/
```

## Notes

- Cityscapes is gated; download manually after registration.
- If diffusion weights are missing, `generate_diffusion_aug.py` will print instructions and exit.
- `scripts/check_topology_loss.py` provides minimal topology loss checks.

#!/usr/bin/env bash

python scripts/generate_diffusion_aug.py --dataset drive --style_name default \
	--style_prompt "retinal imaging, high contrast" --strength 0.4 --num_samples 1 --seed 0

python scripts/generate_diffusion_aug.py --dataset stare --style_name default \
	--style_prompt "retinal imaging, high contrast" --strength 0.4 --num_samples 1 --seed 0

python scripts/generate_diffusion_aug.py --dataset chase --style_name default \
	--style_prompt "retinal imaging, high contrast" --strength 0.4 --num_samples 1 --seed 0

python scripts/generate_diffusion_aug.py --dataset deepglobe --style_name default \
	--style_prompt "satellite imagery, roads, high contrast" --strength 0.5 --num_samples 1 --seed 0

python scripts/generate_diffusion_aug.py --dataset spacenet --style_name default \
	--style_prompt "satellite imagery, roads, high contrast" --strength 0.5 --num_samples 1 --seed 0

python scripts/generate_diffusion_aug.py --dataset ssdd --style_name default \
	--style_prompt "SAR imagery, ships, speckle" --strength 0.5 --num_samples 1 --seed 0

# For GTA5 -> Cityscapes, diffusion is on the source (GTA5):
python scripts/generate_diffusion_aug.py --dataset gta5 --style_name cityscapes \
	--style_prompt "street scene, cityscape, daytime" --strength 0.5 --num_samples 1 --seed 0

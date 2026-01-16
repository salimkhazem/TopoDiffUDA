#!/usr/bin/env bash
set -e

EXP=configs/experiments/main_full_method.yaml
DATASETS=(drive stare chase deepglobe spacenet ssdd gta5_cityscapes)
MODELS=(unet deeplabv3p segformer swin_unet)
SEEDS=(0 1 2)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      OUTDIR_SRC="outputs/runs/source_only/${dataset}/${model}/${seed}"
      if [ ! -f "${OUTDIR_SRC}/summary.csv" ]; then
        python scripts/train.py --config ${EXP} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name source_only --resume
        python scripts/evaluate.py --config ${EXP} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name source_only
      else
        echo "Skipping existing ${OUTDIR_SRC}"
      fi

      OUTDIR_UDA="outputs/runs/main_full_method/${dataset}/${model}/${seed}"
      if [ ! -f "${OUTDIR_UDA}/summary.csv" ]; then
        python scripts/adapt.py --config ${EXP} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name main_full_method --resume
        python scripts/evaluate.py --config ${EXP} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name main_full_method
      else
        echo "Skipping existing ${OUTDIR_UDA}"
      fi
    done
  done
done

#!/usr/bin/env bash
set -e

EXPS=(
  configs/experiments/ablation_no_diffusion.yaml
  configs/experiments/ablation_no_topoloss.yaml
  configs/experiments/ablation_no_pseudofilter.yaml
  configs/experiments/ablation_no_adv.yaml
)

DATASETS=(drive deepglobe ssdd)
MODELS=(unet deeplabv3p)
SEEDS=(0 1 2)

for exp in "${EXPS[@]}"; do
  exp_name=$(basename "${exp}" .yaml)
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        OUTDIR="outputs/runs/${exp_name}/${dataset}/${model}/${seed}"
        if [ -f "${OUTDIR}/summary.csv" ]; then
          echo "Skipping existing ${OUTDIR}"
          continue
        fi
        python scripts/adapt.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed} --resume
        python scripts/evaluate.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed}
      done
    done
  done
done

# Vary topology weight
TOPO_WEIGHTS=(0.0 0.05 0.1 0.2)
for topo in "${TOPO_WEIGHTS[@]}"; do
  exp="configs/experiments/ablation_vary_topoweight.yaml"
  exp_name="ablation_vary_topoweight_t${topo}"
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        OUTDIR="outputs/runs/${exp_name}/${dataset}/${model}/${seed}"
        if [ -f "${OUTDIR}/summary.csv" ]; then
          echo "Skipping existing ${OUTDIR}"
          continue
        fi
        python scripts/adapt.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed} --topo_weight ${topo} --exp_name ${exp_name} --resume
        python scripts/evaluate.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name ${exp_name}
      done
    done
  done
done

# Vary diffusion amount
GEN_AMOUNTS=(0.0 0.25 0.5 1.0)
for gen in "${GEN_AMOUNTS[@]}"; do
  exp="configs/experiments/ablation_vary_gen_amount.yaml"
  exp_name="ablation_vary_gen_amount_p${gen}"
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        OUTDIR="outputs/runs/${exp_name}/${dataset}/${model}/${seed}"
        if [ -f "${OUTDIR}/summary.csv" ]; then
          echo "Skipping existing ${OUTDIR}"
          continue
        fi
        python scripts/adapt.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed} --diffusion_prob ${gen} --exp_name ${exp_name} --resume
        python scripts/evaluate.py --config ${exp} --dataset ${dataset} --model ${model} --seed ${seed} --exp_name ${exp_name}
      done
    done
  done
done

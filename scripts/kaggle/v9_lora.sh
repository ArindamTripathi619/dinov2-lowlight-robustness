#!/bin/bash
export MPLBACKEND=Agg
set -o pipefail
fail=0
for seed in 43 44; do
  d=output/v9_lora/seed${seed}
  mkdir -p "$d"
  python run_lora_simple_colab.py --model dinov2_vitb14 --layers all --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --outdir ${d}/uniform 2>&1 | tee ${d}/uniform.log || fail=1
  python run_lora_simple_colab.py --model dinov2_vitb14 --layers drift --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --drift-csv drift_sev5_low_light.csv \
    --outdir ${d}/drift 2>&1 | tee ${d}/drift.log || fail=1
  python run_lora_simple_colab.py --model dinov2_vitb14 --layers late --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --outdir ${d}/late 2>&1 | tee ${d}/late.log || fail=1
done
exit $fail

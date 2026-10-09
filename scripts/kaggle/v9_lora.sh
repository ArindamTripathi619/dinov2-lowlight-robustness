#!/bin/bash
export MPLBACKEND=Agg
set -o pipefail
mkdir -p output/v9_lora
fail=0
python run_lora_simple_colab.py --model dinov2_vitb14 --layers all --rank 8 \
  --seed 42 --epochs 10 --corruption low_light \
  --outdir output/v9_lora/uniform 2>&1 | tee output/v9_lora/uniform.log || fail=1
python run_lora_simple_colab.py --model dinov2_vitb14 --layers drift --rank 8 \
  --seed 42 --epochs 10 --corruption low_light \
  --drift-csv drift_sev5_low_light.csv \
  --outdir output/v9_lora/drift 2>&1 | tee output/v9_lora/drift.log || fail=1
python run_lora_simple_colab.py --model dinov2_vitb14 --layers late --rank 8 \
  --seed 42 --epochs 10 --corruption low_light \
  --outdir output/v9_lora/late 2>&1 | tee output/v9_lora/late.log || fail=1
exit $fail

#!/bin/bash
# Track 2 v9 allocation-rule ablations (DPA_DESIGN §1.4):
#   btau    — r ∝ β^τ (τ=2, sharpened drift weighting)
#   topk    — hard top-k selection (top-9 drift blocks, equal rank)
#   invbeta — RepSAM-style inverted-β falsification arm
# All three are BUDGET-MATCHED to the drift arm's own rank-sum on the same
# profile, so any performance difference isolates the allocation shape.
# Seeds 42/43/44 = Phase-3 protocol (METHODS §8); the uniform/drift/late
# baselines of the same seeds already exist under output/v9_lora/.
export MPLBACKEND=Agg
set -o pipefail
fail=0
for seed in 42 43 44; do
  d=output/v9_ablation/seed${seed}
  mkdir -p "$d"
  python run_lora_simple_colab.py --model dinov2_vitb14 --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --drift-csv drift_sev5_low_light.csv \
    --layers btau --tau 2 \
    --outdir ${d}/btau 2>&1 | tee ${d}/btau.log || fail=1
  python run_lora_simple_colab.py --model dinov2_vitb14 --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --drift-csv drift_sev5_low_light.csv \
    --layers topk --topk-k 9 \
    --outdir ${d}/topk 2>&1 | tee ${d}/topk.log || fail=1
  python run_lora_simple_colab.py --model dinov2_vitb14 --rank 8 \
    --seed ${seed} --epochs 10 --corruption low_light \
    --drift-csv drift_sev5_low_light.csv \
    --layers invbeta \
    --outdir ${d}/invbeta 2>&1 | tee ${d}/invbeta.log || fail=1
done
exit $fail

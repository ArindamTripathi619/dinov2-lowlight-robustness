#!/bin/bash
# Track 2 v8: ViT-B/14 drift-profile sweep on the T4 (docs/ROADMAP.md section 2).
# Runs from /kaggle/working with CIFAR-10 staged at ./data/cifar-10-batches-py
# and utils/stats_tools/run_drift_proxy/run_drift_profile extracted to cwd.
# The Python driver prints PROFILE_COMPLETE / PROFILE_FAILED based on our exit
# code and the presence of output/vitb_profile/proxy_profiles.csv.
export MPLBACKEND=Agg
mkdir -p output
python run_drift_profile.py \
  --model dinov2_vitb14 \
  --n-images 1000 \
  --severities 1,2,3,4,5 \
  --corruptions low_light,jpeg \
  --batch-size 128 \
  --seed 42 \
  --output output/vitb_profile \
  2>&1 | tee output/vitb_profile.log
exit ${PIPESTATUS[0]}

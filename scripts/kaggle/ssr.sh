#!/bin/bash
# Track 4 SSR S1 smoke kernel (docs/SSR_PLAN.md §7).
#
# S1 = harness + smoke on real ImageNet-val: ResNet-50 (the BN arm) and
# ViT-S/16 (the LN arm), gate verification on a real T4, attachment-point
# check. Both corruptions (low_light + jpeg), severities 1 and 5, seed 42
# (the 3-seed axis is S2's). Arms per SSR_PLAN §6; bn_stats is ResNet-only
# (ViT-S has no BatchNorm); TENT is native-BN on ResNet and the LN
# translation on ViT-S (SSR_PLAN §0.3).
#
# v4 hardening (same set as atlas.sh after the v3 stall):
#   - PYTHONUNBUFFERED=1 + python -u: markers reach the log live.
#   - per-invocation timeout --kill-after so one hung model cannot eat the
#     session; the next invocation still runs.
#   - global deadline guard: never START an invocation that could not finish
#     inside the 12h cap.
#   - SSR_MARK lines (grep-able) around every phase/invocation with
#     wallclock, elapsed seconds, and exit codes.
#   - ImageNet image list cached to ./ssr_cache (one FUSE glob+decode).
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set -o pipefail
fail=0
T_START=$(date +%s)

mark() {
  local line="SSR_MARK ts=$(date -Is) since_start_s=$(( $(date +%s) - T_START )) $*"
  echo "$line"
  mkdir -p output && echo "$line" >> output/ssr_marks.log
}

DEADLINE_S=32400         # 9h since script start
MODEL_LIMIT_S=3600       # per (model, corruption) invocation

IMAGENET_ROOT=""
for cand in /kaggle/input/sautkin/imagenet1kvalid \
            /kaggle/input/datasets/sautkin/imagenet1kvalid; do
  if [ -d "$cand" ]; then IMAGENET_ROOT="$cand"; break; fi
done
if [ -z "$IMAGENET_ROOT" ]; then
  IMAGENET_ROOT=$(find /kaggle/input -maxdepth 4 -type d -name "imagenet1kvalid" 2>/dev/null | head -1)
fi
if [ -z "$IMAGENET_ROOT" ]; then
  mark "event=fatal reason=imagenet_mount_missing"
  echo "FATAL: imagenet1kvalid mount not found under /kaggle/input"
  exit 1
fi
echo "IMAGENET_ROOT=$IMAGENET_ROOT"
mark "event=script_start leg=ssr_s1 root=$IMAGENET_ROOT"

# run_invocation <model-slug> <arms> <profile-csv> <corruption>
# Runs run_ssr.py under timeout, tees its log, records a marker with rc.
run_invocation() {
  local slug=$1 arms=$2 prof=$3 corr=$4
  local safe=$(echo "$slug" | tr ':' '_')
  local d=output/ssr/${safe}_${corr}
  local t0 rc el
  mkdir -p "$d"
  t0=$(date +%s)
  mark "event=run_start model=$slug corruption=$corr dir=$d limit_s=$MODEL_LIMIT_S"
  timeout --kill-after=60s "${MODEL_LIMIT_S}s" python -u run_ssr.py \
    --model "$slug" --profile-csv "$prof" --corruption "$corr" \
    --arms "$arms" --k-fraction 0.25 \
    --n-images 512 --n-est 128 --est-batches 4 --batch-size 64 \
    --severities 1,5 --seed 42 \
    --data-root "$IMAGENET_ROOT" --cache-dir ssr_cache \
    --output "$d" 2>&1 | tee "$d/ssr.log"
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    mark "event=run_timeout model=$slug corruption=$corr duration_s=$el rc=$rc"
    fail=1
  elif [ "$rc" -ne 0 ]; then
    mark "event=run_fail model=$slug corruption=$corr duration_s=$el rc=$rc"
    fail=1
  else
    mark "event=run_done model=$slug corruption=$corr duration_s=$el rc=0"
  fi
}

deadline_ok() {
  local el=$(( $(date +%s) - T_START ))
  if [ "$el" -gt "$DEADLINE_S" ]; then
    mark "event=deadline_hit skip=$1 limit_s=$DEADLINE_S"
    return 1
  fi
  return 0
}

RESNET_ARMS="frozen,ssr_topk,ssr_all,ssr_gate_off,bn_stats,tent,aqr,clahe"
VIT_ARMS="frozen,ssr_topk,ssr_all,ssr_gate_off,tent,aqr,clahe"

for corr in low_light jpeg; do
  if deadline_ok "resnet50_$corr"; then
    run_invocation "resnet50" "$RESNET_ARMS" \
      "output/profiles/resnet50/drift_profile_sev5_${corr}.csv" "$corr"
  else
    fail=1
    break
  fi
  if deadline_ok "vit_small_$corr"; then
    run_invocation "timm:vit_small_patch16_224" "$VIT_ARMS" \
      "output/profiles/vit_small_patch16_224/drift_profile_sev5_${corr}.csv" "$corr"
  else
    fail=1
    break
  fi
done

mark "event=script_done fail=$fail"
exit $fail

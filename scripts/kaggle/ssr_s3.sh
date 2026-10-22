#!/bin/bash
# Track 4 SSR S3 replication (r1) kernel (docs/SSR_PLAN.md §7.2, §10;
# verdict RESEARCH §6.8 — replication readout §6.9).
#
# S2 (kernel ssr-s2, 2026-10-15) found the moment-matching repair's outcome
# flips on attachment scope: ResNet-50 `stream` (k=1, layer4) +4.3 at sev5
# vs `profiled` -32.8; ViT-S `sublayer` -1.4 vs profiled -9.0. That was ONE
# seed (42), ONE corruption (low_light). S3-r1 replicates exactly the two
# rescuing scopes (stream, sublayer) across the seed axis and the jpeg
# family before the verdict is promoted:
#   * 24 replication legs = {stream, sublayer} x {low_light, jpeg} x
#     seeds {42,43,44} x {ResNet-50, ViT-S}, top-25% selection (kcode 25)
#   *  8 k-matched legs @seed42 = the k-confound control (stream is k=1/4
#     and sublayer k=4/16 on ResNet; k=3/12 vs k=6/24 on ViT — match k to
#     isolate location from scope size): ResNet --k 1, ViT --k 3
#   32 legs total, ~50-70 min T4. Arms (frozen · ssr_topk · ssr_gate_off ·
#   tent), severities 1,5, 512 eval / 128 pool — S1/S2 protocol unchanged.
# low_light seed-42 k25 legs are the within-grid S2 parity check (the +4.3
# headline must reproduce across sessions).
# v4 hardening, same set as ssr.sh/ssr_s2.sh: unbuffered markers,
# per-invocation timeout, global deadline guard, SSR_MARK lines, cached
# image list.
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set -o pipefail
fail=0
T_START=$(date +%s)

mark() {
  local line="SSR_MARK ts=$(date -Is) since_start_s=$(( $(date +%s) - T_START )) $*"
  echo "$line"
  mkdir -p output && echo "$line" >> output/ssr_s3_marks.log
}

DEADLINE_S=32400         # 9h since script start
MODEL_LIMIT_S=3600       # per leg

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
mark "event=script_start leg=ssr_s3 root=$IMAGENET_ROOT"

# run_invocation <model-slug> <scope> <corruption> <seed> <kcode>
# kcode 25 = top-25% of the scope's rows (S1/S2 primary policy); any other
# number = absolute --k (the k-matched control). Dir: output/ssr_s3/
# <slug-safe>_<scope>_k<kcode>_<corr>_s<seed>/.
run_invocation() {
  local slug=$1 scope=$2 corr=$3 seed=$4 kcode=$5
  local arms="frozen,ssr_topk,ssr_gate_off,tent"
  local model_dir=${slug#timm:}
  local prof="output/profiles/${model_dir}/drift_profile_sev5_${corr}.csv"
  local kflag="--k-fraction 0.25"
  if [ "$kcode" != "25" ]; then kflag="--k $kcode"; fi
  local safe=$(echo "$slug" | tr ':' '_')
  local d=output/ssr_s3/${safe}_${scope}_k${kcode}_${corr}_s${seed}
  local t0 rc el
  mkdir -p "$d"
  t0=$(date +%s)
  mark "event=run_start model=$slug scope=$scope corruption=$corr seed=$seed k=$kcode dir=$d limit_s=$MODEL_LIMIT_S"
  timeout --kill-after=60s "${MODEL_LIMIT_S}s" python -u run_ssr.py \
    --model "$slug" --profile-csv "$prof" --corruption "$corr" \
    --attach "$scope" --arms "$arms" $kflag \
    --n-images 512 --n-est 128 --est-batches 4 --batch-size 64 \
    --severities 1,5 --seed "$seed" \
    --data-root "$IMAGENET_ROOT" --cache-dir ssr_cache \
    --output "$d" 2>&1 | tee "$d/ssr.log"
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    mark "event=run_timeout model=$slug scope=$scope corruption=$corr seed=$seed k=$kcode duration_s=$el rc=$rc"
    fail=1
  elif [ "$rc" -ne 0 ]; then
    mark "event=run_fail model=$slug scope=$scope corruption=$corr seed=$seed k=$kcode duration_s=$el rc=$rc"
    fail=1
  else
    mark "event=run_done model=$slug scope=$scope corruption=$corr seed=$seed k=$kcode duration_s=$el rc=0"
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

# slug scope corruption seed kcode   (kcode 25 = top-25%; else absolute --k)
LEGS=(
  "resnet50 stream low_light 42 25"
  "resnet50 sublayer low_light 42 25"
  "timm:vit_small_patch16_224 stream low_light 42 25"
  "timm:vit_small_patch16_224 sublayer low_light 42 25"
  "resnet50 stream jpeg 42 25"
  "resnet50 sublayer jpeg 42 25"
  "timm:vit_small_patch16_224 stream jpeg 42 25"
  "timm:vit_small_patch16_224 sublayer jpeg 42 25"
  "resnet50 stream low_light 43 25"
  "resnet50 sublayer low_light 43 25"
  "timm:vit_small_patch16_224 stream low_light 43 25"
  "timm:vit_small_patch16_224 sublayer low_light 43 25"
  "resnet50 stream jpeg 43 25"
  "resnet50 sublayer jpeg 43 25"
  "timm:vit_small_patch16_224 stream jpeg 43 25"
  "timm:vit_small_patch16_224 sublayer jpeg 43 25"
  "resnet50 stream low_light 44 25"
  "resnet50 sublayer low_light 44 25"
  "timm:vit_small_patch16_224 stream low_light 44 25"
  "timm:vit_small_patch16_224 sublayer low_light 44 25"
  "resnet50 stream jpeg 44 25"
  "resnet50 sublayer jpeg 44 25"
  "timm:vit_small_patch16_224 stream jpeg 44 25"
  "timm:vit_small_patch16_224 sublayer jpeg 44 25"
  "resnet50 stream low_light 42 1"
  "resnet50 sublayer low_light 42 1"
  "timm:vit_small_patch16_224 stream low_light 42 3"
  "timm:vit_small_patch16_224 sublayer low_light 42 3"
  "resnet50 stream jpeg 42 1"
  "resnet50 sublayer jpeg 42 1"
  "timm:vit_small_patch16_224 stream jpeg 42 3"
  "timm:vit_small_patch16_224 sublayer jpeg 42 3"
)

for leg in "${LEGS[@]}"; do
  read -r slug scope corr seed kcode <<< "$leg"
  if deadline_ok "${slug}_${scope}_${corr}_s${seed}_k${kcode}"; then
    run_invocation "$slug" "$scope" "$corr" "$seed" "$kcode"
  else
    fail=1
    break
  fi
done

mark "event=script_done fail=$fail legs=${#LEGS[@]}"
exit $fail

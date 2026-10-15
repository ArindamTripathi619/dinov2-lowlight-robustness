#!/bin/bash
# Track 4 SSR S2 attachment-point probe kernel (docs/SSR_PLAN.md §7.2, §10).
#
# S2 answers the S1 open question: was moment matching harmful because the
# mechanism is wrong, or because it was attached post-norm (LN/BN normalize
# channel moments away at exactly the point S1 repaired)? Same runner, same
# protocol as S1 (seed 42, severities 1 and 5, 512 eval images, 128-image
# pool, top-25% drift selection, gate + identity check); only the
# attachment scope (--attach) varies:
#   profiled  S1's atlas-hooked module set — within-script control; frozen
#             and ssr_topk must reproduce the S1 low_light numbers
#   sublayer  ViT attn/mlp outputs / ResNet bottleneck outputs (pre-norm
#             residual stream — where the atlas profiles point, §2)
#   stream    ViT blocks.N / ResNet stage outputs (coarser stream points)
#   norm      ViT post-LN norm1/norm2 outputs — the S1 attachment class
#             re-run alone (ViT only; ResNet's norm modules have no
#             profiled drift ranks to borrow, §7.2)
# Corruption: low_light only — S1's maximal-harm family and the §2
# luminance-hypothesis family. Arms: frozen, ssr_topk, ssr_gate_off, tent
# (tent is the family control: adaptation-only, unaffected by attachment).
# v4 hardening, same set as ssr.sh: unbuffered markers, per-invocation
# timeout, global deadline guard, SSR_MARK lines, cached image list.
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set -o pipefail
fail=0
T_START=$(date +%s)

mark() {
  local line="SSR_MARK ts=$(date -Is) since_start_s=$(( $(date +%s) - T_START )) $*"
  echo "$line"
  mkdir -p output && echo "$line" >> output/ssr_s2_marks.log
}

DEADLINE_S=32400         # 9h since script start
MODEL_LIMIT_S=3600       # per (model, scope) invocation

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
mark "event=script_start leg=ssr_s2 root=$IMAGENET_ROOT"

# run_invocation <model-slug> <attach-scope> <arms> <profile-csv>
# One (model, scope) leg: run_ssr.py under timeout, tees its log, records a
# marker with rc. Corruption is fixed to low_light (the S2 probe family).
run_invocation() {
  local slug=$1 scope=$2 arms=$3 prof=$4
  local safe=$(echo "$slug" | tr ':' '_')
  local d=output/ssr_s2/${safe}_${scope}_low_light
  local t0 rc el
  mkdir -p "$d"
  t0=$(date +%s)
  mark "event=run_start model=$slug scope=$scope dir=$d limit_s=$MODEL_LIMIT_S"
  timeout --kill-after=60s "${MODEL_LIMIT_S}s" python -u run_ssr.py \
    --model "$slug" --profile-csv "$prof" --corruption low_light \
    --attach "$scope" --arms "$arms" --k-fraction 0.25 \
    --n-images 512 --n-est 128 --est-batches 4 --batch-size 64 \
    --severities 1,5 --seed 42 \
    --data-root "$IMAGENET_ROOT" --cache-dir ssr_cache \
    --output "$d" 2>&1 | tee "$d/ssr.log"
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    mark "event=run_timeout model=$slug scope=$scope duration_s=$el rc=$rc"
    fail=1
  elif [ "$rc" -ne 0 ]; then
    mark "event=run_fail model=$slug scope=$scope duration_s=$el rc=$rc"
    fail=1
  else
    mark "event=run_done model=$slug scope=$scope duration_s=$el rc=0"
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

ARMS="frozen,ssr_topk,ssr_gate_off,tent"
RN_PROFILE="output/profiles/resnet50/drift_profile_sev5_low_light.csv"
VIT_PROFILE="output/profiles/vit_small_patch16_224/drift_profile_sev5_low_light.csv"

for scope in profiled sublayer stream; do
  if deadline_ok "resnet50_$scope"; then
    run_invocation "resnet50" "$scope" "$ARMS" "$RN_PROFILE"
  else
    fail=1
    break
  fi
done

for scope in profiled sublayer stream norm; do
  if deadline_ok "vit_small_$scope"; then
    run_invocation "timm:vit_small_patch16_224" "$scope" "$ARMS" "$VIT_PROFILE"
  else
    fail=1
    break
  fi
done

mark "event=script_done fail=$fail"
exit $fail

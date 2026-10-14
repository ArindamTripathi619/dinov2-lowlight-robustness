#!/bin/bash
# Track 3 cross-family drift atlas kernel (docs/ROADMAP.md §2 Track 3).
#
# Atlas of record: 6 models x {low_light, jpeg} x severities 1-5 on a shared
# ImageNet-val subset (n=2000, seeded 42, sautkin/imagenet1kvalid mount).
# Consistency leg: resnet50 + convnext_tiny on CIFAR-10 test n=1000 (the
# legacy image axis) to anchor the atlas rows against the Track 1/2 artifacts.
#
# vmamba stretch is dead: timm 1.0.30 ships no vmamba models (needs the
# separate ssm package) — recorded in the Track 3 docs instead.
#
# v4 hardening (after the drift-atlas v3 run stalled with no observable
# progress past Kaggle's 12h session cap):
#   - PYTHONUNBUFFERED=1 + `python -u`: every marker reaches the log live
#     and survives SIGTERM/SIGKILL (no lost block-buffered tail).
#   - per-model `timeout --kill-after` so one hung model (FUSE stall,
#     download backoff) cannot eat the whole session; next model still runs.
#   - global deadline guard: never START a model that could not finish
#     inside the 12h cap; skip with a marker instead.
#   - ATLAS_MARK lines (grep-able) around every phase/model with wallclock,
#     elapsed seconds, and exit codes — even a killed run leaves a complete
#     audit trail in each profile.log.
#   - ImageNet image list cached to ./atlas_cache (single FUSE glob+decode
#     instead of one per model invocation).
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set -o pipefail
fail=0
T_START=$(date +%s)

mark() {
  local line="ATLAS_MARK ts=$(date -Is) since_start_s=$(( $(date +%s) - T_START )) $*"
  echo "$line"
  # Mirror into an output file: Kaggle captures relative-path outputs (proven
  # on the completed sibling kernels), so the audit trail survives even if
  # the logs endpoint returns nothing.
  mkdir -p output && echo "$line" >> output/atlas_marks.log
}

# Never start a model that could not finish well inside Kaggle's 12h cap.
DEADLINE_S=30600          # 8.5h since script start
IM_LIMIT_S=4500           # per ImageNet model (load + 11 passes + stats + plots)
CIFAR_LIMIT_S=2400        # per CIFAR consistency model

# run_model <limit_s> <output_dir> <python-args...>
# Runs the profiler under timeout, tees its log, then records a marker with
# rc + elapsed. rc 124/137 = timeout killed it (treated as failure, loop
# continues to the next model).
run_model() {
  local limit=$1 d=$2; shift 2
  local t0 rc el
  mkdir -p "$d"   # tee target must exist: a missing dir would make tee's rc
                   # mask timeout's 124 under pipefail and mislabel a hang
  t0=$(date +%s)
  mark "event=model_start model_dir=$d limit_s=$limit"
  timeout --kill-after=60s "${limit}s" python -u "$@" 2>&1 | tee "$d/profile.log"
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    mark "event=model_timeout model_dir=$d duration_s=$el rc=$rc"
    fail=1
  elif [ "$rc" -ne 0 ]; then
    mark "event=model_fail model_dir=$d duration_s=$el rc=$rc"
    fail=1
  else
    mark "event=model_done model_dir=$d duration_s=$el rc=0"
  fi
}

# deadline_ok <model> -> prints marker + returns 1 when past the deadline
deadline_ok() {
  local el=$(( $(date +%s) - T_START ))
  if [ "$el" -gt "$DEADLINE_S" ]; then
    mark "event=deadline_hit skip=$1 limit_s=$DEADLINE_S"
    return 1
  fi
  return 0
}

# Resolve the ImageNet-val mount: Kaggle layouts differ between
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>.
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
echo "IMAGENET_ROOT=$IMAGENET_ROOT ($(find "$IMAGENET_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l) class folders)"
mark "event=script_start leg=imagenet root=$IMAGENET_ROOT"

deadline_hit=0
for m in resnet50 timm:convnext_tiny timm:vit_small_patch16_224 \
         timm:swin_tiny_patch4_window7_224 timm:deit_small_patch16_224 \
         dinov2_vitb14; do
  slug=$(echo "$m" | sed 's/^timm://')
  d=output/profiles/${slug}
  if ! deadline_ok "$m"; then deadline_hit=1; fail=1; break; fi
  mkdir -p "$d"
  run_model "$IM_LIMIT_S" "$d" run_drift_profile.py --model "$m" --dataset imagenet \
    --data-root "$IMAGENET_ROOT" --n-images 2000 \
    --severities 1,2,3,4,5 --corruptions low_light,jpeg \
    --output "$d" --cache-dir atlas_cache
done

# CIFAR-10 consistency leg (same protocol as the committed Track 1/2 rows)
if [ "$deadline_hit" -eq 0 ]; then
  mark "event=leg_start leg=cifar10"
  for m in resnet50 timm:convnext_tiny; do
    slug=$(echo "$m" | sed 's/^timm://')
    d=output/profiles/cifar10_${slug}
    if ! deadline_ok "$m"; then deadline_hit=1; fail=1; break; fi
    mkdir -p "$d"
    run_model "$CIFAR_LIMIT_S" "$d" run_drift_profile.py --model "$m" --dataset cifar10 \
      --n-images 1000 --severities 1,5 --corruptions low_light,jpeg \
      --output "$d"
  done
else
  mark "event=legs_skipped leg=cifar10 reason=deadline"
fi

mark "event=script_done fail=$fail"
exit $fail

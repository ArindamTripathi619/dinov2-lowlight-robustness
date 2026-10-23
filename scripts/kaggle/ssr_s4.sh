#!/bin/bash
# Track 4 SSR S4 6-model grid kernel (docs/SSR_PLAN.md §7.3, §6;
# verdict RESEARCH §6.8/§6.10 — replication readout).
#
# S3 (kernel ssr-s3) established on {ResNet-50, ViT-S} that the moment-matching
# repair's outcome flips on attachment scope: ResNet `stream` (layer4) rescues
# (+4.3 at sev5) vs `profiled` -32.8; ViT-S `sublayer` rescues vs `profiled`
# -9.0. S4 asks whether that verdict generalizes to the four remaining atlas
# models (ConvNeXt-T, Swin-T, DeiT-S, DINOv2-ViT-B/14):
#   * ConvNeXt/Swin (hierarchical stage models) get new stream=`stages.N`/
#     `layers.N` and sublayer=`stages.N.blocks|downsample` scope patterns
#     (src/ssr.py; profile stops at depth 2 so within-stage granularity is the
#     blocks-container + downsample). DeiT/DINOv2 reuse the ViT patterns and
#     additionally run the `norm` scope (they carry attn/mlp rows to alias).
#   * DINOv2 has no ImageNet head -> CIFAR-10 + frozen-probe protocol
#     (SSR_PLAN §6): linear probe on clean train embeddings, frozen; SSR still
#     calibrates/estimates/gates on the backbone.
#   * profiled-dedup control per model: the output-identity dedup (run_ssr.py)
#     now drops same-tensor container/child pairs (blocks/blocks.11, stages/
#     stages.3) so the affine applies once per tensor — the S1 profiled-arm
#     caveat resolved (docs/METHODS.md).
# 68 legs = {stream,sublayer}(/norm for DeiT+DINOv2) x {low_light,jpeg} x
# seeds{42,43,44} at k=25 (primary) + profiled-dedup x {low_light,jpeg} @
# seed42 (control). Arms frozen · ssr_topk · ssr_gate_off · tent; severities
# 1,5; 512 eval / 128 pool — S1/S2/S3 protocol unchanged.
# v4 hardening, same as ssr.sh/ssr_s2.sh/ssr_s3.sh: unbuffered markers,
# per-invocation timeout, global deadline guard, SSR_MARK lines, cached
# image list (ImageNet legs) / torchvision CIFAR-10 download (dinov2 legs).
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
set -o pipefail
fail=0
T_START=$(date +%s)

mark() {
  local line="SSR_MARK ts=$(date -Is) since_start_s=$(( $(date +%s) - T_START )) $*"
  echo "$line"
  mkdir -p output && echo "$line" >> output/ssr_s4_marks.log
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
mark "event=script_start leg=ssr_s4 root=$IMAGENET_ROOT"

# run_invocation <model-slug> <scope> <corruption> <seed> <kcode>
# kcode 25 = top-25% of the scope's rows (S1/S2/S3/S4 primary policy).
# Dir: output/ssr_s4/<slug-safe>_<scope>_k<kcode>_<corr>_s<seed>/.
# DINOv2 legs run the CIFAR-10 frozen-probe protocol (dataset auto-resolved
# inside run_ssr.py from the dinov2 prefix); timm legs use the ImageNet head.
run_invocation() {
  local slug=$1 scope=$2 corr=$3 seed=$4 kcode=$5
  local arms="frozen,ssr_topk,ssr_gate_off,tent"
  local model_dir=${slug#timm:}
  local prof="output/profiles/${model_dir}/drift_profile_sev5_${corr}.csv"
  local kflag="--k-fraction 0.25"
  if [ "$kcode" != "25" ]; then kflag="--k $kcode"; fi
  local safe=$(echo "$slug" | tr ':' '_')
  local d=output/ssr_s4/${safe}_${scope}_k${kcode}_${corr}_s${seed}
  local ds_flags="--data-root $IMAGENET_ROOT --cache-dir ssr_cache"
  if [[ "$slug" == dinov2* ]]; then ds_flags="--data-root ./data"; fi
  local t0 rc el
  mkdir -p "$d"
  t0=$(date +%s)
  mark "event=run_start model=$slug scope=$scope corruption=$corr seed=$seed k=$kcode dir=$d limit_s=$MODEL_LIMIT_S"
  timeout --kill-after=60s "${MODEL_LIMIT_S}s" python -u run_ssr.py \
    --model "$slug" --profile-csv "$prof" --corruption "$corr" \
    --attach "$scope" --arms "$arms" $kflag \
    --n-images 512 --n-est 128 --est-batches 4 --batch-size 64 \
    --severities 1,5 --seed "$seed" \
    $ds_flags \
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

# slug scope corruption seed kcode   (kcode 25 = top-25%)
LEGS=(
  "timm:convnext_tiny stream low_light 42 25"
  "timm:convnext_tiny sublayer low_light 42 25"
  "timm:swin_tiny_patch4_window7_224 stream low_light 42 25"
  "timm:swin_tiny_patch4_window7_224 sublayer low_light 42 25"
  "timm:deit_small_patch16_224 stream low_light 42 25"
  "timm:deit_small_patch16_224 sublayer low_light 42 25"
  "timm:deit_small_patch16_224 norm low_light 42 25"
  "dinov2_vitb14 stream low_light 42 25"
  "dinov2_vitb14 sublayer low_light 42 25"
  "dinov2_vitb14 norm low_light 42 25"
  "timm:convnext_tiny stream jpeg 42 25"
  "timm:convnext_tiny sublayer jpeg 42 25"
  "timm:swin_tiny_patch4_window7_224 stream jpeg 42 25"
  "timm:swin_tiny_patch4_window7_224 sublayer jpeg 42 25"
  "timm:deit_small_patch16_224 stream jpeg 42 25"
  "timm:deit_small_patch16_224 sublayer jpeg 42 25"
  "timm:deit_small_patch16_224 norm jpeg 42 25"
  "dinov2_vitb14 stream jpeg 42 25"
  "dinov2_vitb14 sublayer jpeg 42 25"
  "dinov2_vitb14 norm jpeg 42 25"
  "timm:convnext_tiny stream low_light 43 25"
  "timm:convnext_tiny sublayer low_light 43 25"
  "timm:swin_tiny_patch4_window7_224 stream low_light 43 25"
  "timm:swin_tiny_patch4_window7_224 sublayer low_light 43 25"
  "timm:deit_small_patch16_224 stream low_light 43 25"
  "timm:deit_small_patch16_224 sublayer low_light 43 25"
  "timm:deit_small_patch16_224 norm low_light 43 25"
  "dinov2_vitb14 stream low_light 43 25"
  "dinov2_vitb14 sublayer low_light 43 25"
  "dinov2_vitb14 norm low_light 43 25"
  "timm:convnext_tiny stream jpeg 43 25"
  "timm:convnext_tiny sublayer jpeg 43 25"
  "timm:swin_tiny_patch4_window7_224 stream jpeg 43 25"
  "timm:swin_tiny_patch4_window7_224 sublayer jpeg 43 25"
  "timm:deit_small_patch16_224 stream jpeg 43 25"
  "timm:deit_small_patch16_224 sublayer jpeg 43 25"
  "timm:deit_small_patch16_224 norm jpeg 43 25"
  "dinov2_vitb14 stream jpeg 43 25"
  "dinov2_vitb14 sublayer jpeg 43 25"
  "dinov2_vitb14 norm jpeg 43 25"
  "timm:convnext_tiny stream low_light 44 25"
  "timm:convnext_tiny sublayer low_light 44 25"
  "timm:swin_tiny_patch4_window7_224 stream low_light 44 25"
  "timm:swin_tiny_patch4_window7_224 sublayer low_light 44 25"
  "timm:deit_small_patch16_224 stream low_light 44 25"
  "timm:deit_small_patch16_224 sublayer low_light 44 25"
  "timm:deit_small_patch16_224 norm low_light 44 25"
  "dinov2_vitb14 stream low_light 44 25"
  "dinov2_vitb14 sublayer low_light 44 25"
  "dinov2_vitb14 norm low_light 44 25"
  "timm:convnext_tiny stream jpeg 44 25"
  "timm:convnext_tiny sublayer jpeg 44 25"
  "timm:swin_tiny_patch4_window7_224 stream jpeg 44 25"
  "timm:swin_tiny_patch4_window7_224 sublayer jpeg 44 25"
  "timm:deit_small_patch16_224 stream jpeg 44 25"
  "timm:deit_small_patch16_224 sublayer jpeg 44 25"
  "timm:deit_small_patch16_224 norm jpeg 44 25"
  "dinov2_vitb14 stream jpeg 44 25"
  "dinov2_vitb14 sublayer jpeg 44 25"
  "dinov2_vitb14 norm jpeg 44 25"
  "timm:convnext_tiny profiled low_light 42 25"
  "timm:swin_tiny_patch4_window7_224 profiled low_light 42 25"
  "timm:deit_small_patch16_224 profiled low_light 42 25"
  "dinov2_vitb14 profiled low_light 42 25"
  "timm:convnext_tiny profiled jpeg 42 25"
  "timm:swin_tiny_patch4_window7_224 profiled jpeg 42 25"
  "timm:deit_small_patch16_224 profiled jpeg 42 25"
  "dinov2_vitb14 profiled jpeg 42 25"
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

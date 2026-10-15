"""
Selective statistic recalibration (SSR) runner — Track 4 (docs/SSR_PLAN.md)
==========================================================================

Label-free, backprop-free test-time recalibration driven by the Track 3
drift atlas: the same severity-5 CKA profile that allocated LoRA rank in
Track 2 selects which module outputs SSR recalibrates here (DPA_DESIGN §1.4
— one measurement, two consumers). Selection is per (model, corruption
family) straight from output/profiles/<model>/drift_profile_sev5_<corr>.csv.

Protocol (SSR_PLAN §5/§6):
  * clean calibration on a pool disjoint from eval images -> per-module
    channel moments (mu_c, sigma_c) + the gate threshold tau (fit on clean
    per-batch distances);
  * per (corruption, severity): estimate moments from B unlabeled shifted
    batches (also disjoint from eval), compute the aggregated shift
    distance, and decide the gate: closed -> bypass entirely (bit-identical
    by construction, src.ssr.recalibrate returns the input object);
  * eval: top-1 of the frozen ImageNet head on the shifted eval images.

Arms (SSR_PLAN §6):
  frozen         no recalibration, reference
  ssr_topk       gate + moment affine on the top-k drift modules (primary)
  ssr_all        gate + moment affine on every profiled module
  ssr_gate_off   moment affine on top-k, applied unconditionally (gate
                 ablation: measures clean-side damage of always-on SSR)
  bn_stats       Schneider-style BN running-stat replacement from the
                 estimation batches (ResNet-50 only; both scopes via
                 --bn-scope all|topk)
  tent           TENT (Wang et al. ICLR'21): entropy minimization on the
                 norm affine params; native on BN (ResNet-50), LN
                 translation elsewhere (SSR_PLAN §0.3, clearly labeled)
  aqr            AQR (arXiv:2511.03148): per-channel piecewise-linear
                 quantile transport of module outputs onto stored source
                 percentiles; same selection as ssr_topk (mechanism
                 ablation: moments vs quantiles, DPA_DESIGN §4)
  clahe          CLAHE enhancement front-end (src.corruptions.clahe_enhance),
                 frozen model behind it

The identity gate is verified every run (SSR_PLAN §5, claim C2): SSR armed
on clean eval images must produce bit-identical logits to frozen.

Usage (S1 smoke, SSR_PLAN §7):
  python3 run_ssr.py --model resnet50 --arms frozen,ssr_topk,ssr_all,bn_stats,tent \
      --profile-csv output/profiles/resnet50/drift_profile_sev5_low_light.csv \
      --corruption low_light --n-images 512 --output output/ssr/resnet50
  python3 run_ssr.py --model timm:vit_small_patch16_224 \
      --arms frozen,ssr_topk,tent,aqr --profile-csv .../vit_small_patch16_224/...csv

Usage (S2 attachment-point probe, SSR_PLAN §7.2): same flags plus --attach:
  python3 run_ssr.py --attach sublayer --arms frozen,ssr_topk,ssr_gate_off,tent \
      --model timm:vit_small_patch16_224 --profile-csv .../drift_profile_sev5_low_light.csv \
      --corruption low_light --output output/ssr_s2/timm_vit_small_patch16_224_sublayer_low_light
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from utils import DEVICE, get_corruption, load_imagenet_subset
import run_drift_profile as rdp_main
import run_drift_proxy as rdp
from src.aqr import apply_aqr, fit_channel_quantiles
from src.ssr import (SCOPE_CHOICES, aggregate_distances, channel_axis_for, fit_gate,
                     gate_open, looks_channels_last, map_scope_rows, module_in_scope,
                     moment_affine, moment_distance, pool_stats, select_topk,
                     select_topk_fraction)
from tools.build_atlas import read_profile_csv

ALL_ARMS = ["frozen", "ssr_topk", "ssr_all", "ssr_gate_off", "bn_stats", "tent",
            "aqr", "clahe"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Selective statistic recalibration (Track 4 SSR runner)")
    p.add_argument("--model", default="resnet50",
                   help="resnet50 (torchvision, has BN) or timm:<name> (LN models)")
    p.add_argument("--profile-csv", required=True,
                   help="severity-5 atlas CSV whose cka_unbiased_drop ranks the "
                        "modules (output/profiles/<model>/drift_profile_sev5_<corr>.csv)")
    p.add_argument("--corruption", required=True,
                   help="corruption family this profile was measured under; gates "
                        "arm selection semantics (one runner call per family)")
    p.add_argument("--arms", default="frozen,ssr_topk,ssr_all,ssr_gate_off",
                   help=f"comma-separated subset of {ALL_ARMS}")
    p.add_argument("--k", type=int, default=None,
                   help="override the number of top-k modules (default: top 25%% "
                        "of profiled modules, SSR_PLAN §4)")
    p.add_argument("--k-fraction", type=float, default=0.25,
                   help="top-k policy when --k is not given")
    p.add_argument("--bn-scope", default="all", choices=["all", "topk"],
                   help="BN-stat replacement scope (ResNet-50 arm)")
    p.add_argument("--attach", default="profiled", choices=list(SCOPE_CHOICES),
                   help="S2 attachment scope (SSR_PLAN §7.2): profiled = the S1 "
                        "atlas-hooked module set; stream = residual-stream outputs "
                        "(ViT blocks.N, ResNet stage outputs); sublayer = attn/mlp "
                        "outputs (ViT) or bottleneck outputs (ResNet); norm = "
                        "post-LN norm1/norm2 outputs (ViT only)")
    p.add_argument("--n-images", type=int, default=512,
                   help="eval images (the tail of the seeded draw; the head feeds "
                        "the calibration/estimation pool)")
    p.add_argument("--n-est", type=int, default=128,
                   help="calibration/estimation pool size, disjoint from eval "
                        "(SSR_PLAN §6: B batches drawn from this pool)")
    p.add_argument("--est-batches", type=int, default=4,
                   help="number of unlabeled estimation batches per condition")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--seed", type=int, default=42,
                   help="estimation-pool draw seed (the S2 3-seed axis)")
    p.add_argument("--data-root", default="./data/imagenet-val")
    p.add_argument("--cache-dir", default=None,
                   help="image-list cache dir (same semantics as run_drift_profile)")
    p.add_argument("--gate-quantile", type=float, default=1.0,
                   help="clean-distance quantile for tau (1.0 = max, strictest)")
    p.add_argument("--gate-margin", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=4.0,
                   help="moment-affine scale clip (src.ssr.DEFAULT_CLIP)")
    p.add_argument("--tent-lr", type=float, default=1e-3)
    p.add_argument("--tent-steps", type=int, default=1,
                   help="entropy-min steps per estimation batch")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--vit-token", default="cls", choices=["cls", "mean"])
    p.add_argument("--output", default="output/ssr")
    return p.parse_args()


def select_modules_for_hooks(model, max_depth=2, attach="profiled"):
    """Profiled modules (same selection the atlas used) as (name, module) in
    forward order — or, for an S2 attachment scope, every module in that
    scope (norm layers are leaves, so the depth-bounded container walk
    cannot see them)."""
    if attach == "profiled":
        return rdp.select_profiled_modules(model, max_depth=max_depth)
    return {name: module for name, module in model.named_modules()
            if name and module_in_scope(name, attach)}


def probe_channel_axes(model, modules, batch, device, token_mode="cls"):
    """One clean forward to (a) drop modules that never fire and (b) resolve
    each firing module's channel axis from its first output shape."""
    axes = {}
    handles = []
    for name, module in modules.items():
        def mk(name):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(t):
                    return
                if name not in axes:
                    cl = looks_channels_last(tuple(t.shape)) if t.dim() == 4 else False
                    axes[name] = channel_axis_for(tuple(t.shape), channels_last=cl)
            return hook
        handles.append(module.register_forward_hook(mk(name)))
    with torch.no_grad():
        rdp.forward_pass(model, batch, modules, len(batch), device,
                         token_mode=token_mode)
    for h in handles:
        h.remove()
    return {n: modules[n] for n in axes}, axes


class MomentRecorder:
    """Forward hooks accumulating per-batch channel moments for a module set."""

    def __init__(self, modules, axes):
        self.axes = axes
        self.pairs = {name: [] for name in modules}
        self.handles = [
            module.register_forward_hook(self._mk(name))
            for name, module in modules.items()
        ]

    def _mk(self, name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            arr = t.detach().to("cpu", torch.float32).numpy()
            self.pairs[name].append(channel_stats_np(arr, self.axes[name]))
        return hook

    def pooled(self, n_images):
        out = {}
        for name, pairs in self.pairs.items():
            if not pairs:
                continue
            weights = [p[2] for p in pairs]
            out[name] = pool_stats([(p[0], p[1]) for p in pairs], weights=weights)
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def channel_stats_np(arr, channel_axis):
    from src.ssr import channel_stats
    mean, std = channel_stats(arr, channel_axis=channel_axis)
    n = int(np.prod([s for i, s in enumerate(arr.shape) if i != channel_axis]))
    return mean, std, n


def shift_distances(clean_stats, est_stats, names):
    return {n: moment_distance(clean_stats[n], est_stats[n]) for n in names}


class RecalibrationHooks:
    """Forward hooks applying the SSR affine (or an AQR map) to module outputs
    when armed. Disarmed hooks pass the output through untouched, so attaching
    them cannot perturb the frozen path (bit-identity contract)."""

    def __init__(self, transforms, axes):
        self.transforms = {}
        self.axes = axes
        self.handles = []
        for name, fn in transforms.items():
            self.transforms[name] = fn
        self.armed = False

    def attach(self, modules):
        for name in self.transforms:
            if name in modules:
                self.handles.append(
                    modules[name].register_forward_hook(self._mk(name)))
        return self

    def _mk(self, name):
        def hook(_m, _inp, out):
            if not self.armed:
                return None
            fn = self.transforms[name]
            if isinstance(out, (tuple, list)):
                for i, t in enumerate(out):
                    if torch.is_tensor(t):
                        new = fn(t, self.axes[name])
                        return (new,) + tuple(out[1:])
                return None
            if torch.is_tensor(out):
                return fn(out, self.axes[name])
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def affine_transform(a_np, b_np, clip):
    def fn(t, axis):
        a = np.asarray(a_np, dtype=np.float32).reshape(-1)
        b = np.asarray(b_np, dtype=np.float32).reshape(-1)
        shape = [1] * t.dim()
        shape[axis] = -1
        return (t * torch.from_numpy(a).reshape(shape).to(t.device)
                + torch.from_numpy(b).reshape(shape).to(t.device)).to(t.dtype)
    return fn


def aqr_transform(source_q, target_q):
    def fn(t, axis):
        arr = t.detach().to("cpu", torch.float32).numpy()
        out = apply_aqr(arr, source_q, target_q, channel_axis=axis)
        return torch.from_numpy(out.astype(np.float32)).to(t.device).to(t.dtype)
    return fn


@torch.no_grad()
def evaluate(model, images, labels, batch_size, device, token_mode="cls",
             preprocess=None):
    """Top-1 of the model's own head over the eval images."""
    preprocess = preprocess or rdp.PREPROCESS
    correct, n = 0, 0
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        logits = model(tensors)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("out"))
        pred = logits.argmax(dim=1).cpu().numpy()
        correct += int((pred == labels[i:i + batch_size]).sum())
        n += len(batch)
    return correct / n


@torch.no_grad()
def corrupt_pool(images, fn, severity, rng_seed):
    rng = np.random.default_rng(rng_seed)
    return [fn(img, severity, rng=rng) for img in images]


def run_forward(model, images, modules, axes, batch_size, device,
                token_mode="cls", transforms=None):
    hooks = None
    if transforms:
        hooks = RecalibrationHooks(transforms, axes).attach(modules)
    try:
        with torch.no_grad():
            rdp.forward_pass(model, images, modules, batch_size, device,
                             token_mode=token_mode)
    finally:
        if hooks is not None:
            hooks.remove()


def collect_stats(model, images, modules, axes, batch_size, device,
                  token_mode="cls", transforms=None):
    """One forward pass; returns pooled per-module (mean, std) (+ optional
    per-module activation arrays for the AQR arm)."""
    rec = MomentRecorder(modules, axes)
    hooks = None
    if transforms:
        hooks = RecalibrationHooks(transforms, axes).attach(modules)
    try:
        with torch.no_grad():
            rdp.forward_pass(model, images, modules, batch_size, device,
                             token_mode=token_mode)
    finally:
        rec.remove()
        if hooks is not None:
            hooks.remove()
    return rec.pooled(len(images))


def collect_activations(model, images, modules, axes, batch_size, device,
                        token_mode="cls", max_samples=4096):
    """Per-module activations (samples x channels, channel-last) for quantile
    fitting. Capped at max_samples rows per module."""
    store = {name: [] for name in modules}
    counts = {name: 0 for name in modules}

    def mk(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or counts[name] >= max_samples:
                return
            arr = t.detach().to("cpu", torch.float32).numpy()
            moved = np.moveaxis(arr, axes[name], -1).reshape(-1, arr.shape[axes[name]])
            take = min(len(moved), max_samples - counts[name])
            store[name].append(moved[:take])
            counts[name] += take
        return hook

    handles = [module.register_forward_hook(mk(name)) for name, module in modules.items()]
    try:
        with torch.no_grad():
            rdp.forward_pass(model, images, modules, batch_size, device,
                             token_mode=token_mode)
    finally:
        for h in handles:
            h.remove()
    return {n: np.concatenate(v, axis=0) for n, v in store.items() if v}


def norm_affine_params(model, kind):
    import torch.nn as nn
    cls = nn.BatchNorm2d if kind == "bn" else nn.LayerNorm
    mods = [m for m in model.modules() if isinstance(m, cls)]
    params = []
    for m in mods:
        if getattr(m, "weight", None) is not None and m.weight.requires_grad is not False:
            params += [p for p in m.parameters()]
    return mods, params


def entropy_loss(logits):
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    return -(p * logp).sum(dim=1).mean()


def tent_adapt(model, images, batch_size, device, kind, lr, steps):
    """One TENT adaptation pass over the unlabeled estimation batches."""
    import torch.nn as nn
    mods, params = norm_affine_params(model, kind)
    if not params:
        return False
    snapshot = [(p, p.detach().clone(), p.requires_grad) for p in params]
    was_training = model.training
    for p in params:
        p.requires_grad_(True)
    if kind == "bn":
        for m in mods:
            m.train()
    opt = torch.optim.Adam(params, lr=lr)
    try:
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([rdp.PREPROCESS(img) for img in batch]).to(device)
            for _ in range(steps):
                opt.zero_grad()
                loss = entropy_loss(model(tensors))
                loss.backward()
                opt.step()
    finally:
        for p, snap, req in snapshot:
            p.data.copy_(snap)
            p.requires_grad_(req)
        for m in mods:
            m.eval()
        model.train(was_training)
    return True


def bn_stat_replace(model, est_images, batch_size, device, scope_modules=None):
    """Schneider-style: set BN running stats from the estimation batches'
    input moments. Returns a restore callable."""
    import torch.nn as nn
    bns = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if scope_modules is not None and not any(
                    name == s or name.startswith(s + ".") for s in scope_modules):
                continue
            bns.append((name, m))
    saved = {n: (m.running_mean.clone(), m.running_var.clone(),
                 m.num_batches_tracked.clone()) for n, m in bns}
    inputs = {n: [] for n, _ in bns}
    handles = []
    for n, m in bns:
        def mk(n, m):
            def pre(_m, args, _kwargs=None):
                x = args[0]
                arr = x.detach().to("cpu", torch.float32).numpy()
                mean, std = arr.mean(axis=(0, 2, 3)), arr.std(axis=(0, 2, 3))
                inputs[n].append((mean, std, arr.shape[0]))
            return pre
        handles.append(m.register_forward_pre_hook(mk(n, m)))
    with torch.no_grad():
        for i in range(0, len(est_images), batch_size):
            batch = est_images[i:i + batch_size]
            tensors = torch.stack([rdp.PREPROCESS(img) for img in batch]).to(device)
            model(tensors)
    for h in handles:
        h.remove()
    with torch.no_grad():
        for n, m in bns:
            if not inputs[n]:
                continue
            mean, std = pool_stats([(p[0], p[1]) for p in inputs[n]],
                                   weights=[p[2] for p in inputs[n]])
            m.running_mean.copy_(torch.from_numpy(mean.astype(np.float32)).to(m.running_mean.device))
            m.running_var.copy_(torch.from_numpy((std ** 2).astype(np.float32)).to(m.running_var.device))
            m.num_batches_tracked.zero_()

    def restore():
        with torch.no_grad():
            for n, m in bns:
                rm, rv, nb = saved[n]
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)
    return restore


def main():
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; supported: {ALL_ARMS}")
    severities = sorted({int(s) for s in args.severities.split(",") if s.strip()})
    if not severities or any(s < 1 or s > 5 for s in severities):
        raise ValueError(f"--severities must be in 1..5, got {severities}")
    if args.model.startswith("dinov2"):
        raise NotImplementedError(
            "DINOv2 has no ImageNet head; its SSR leg runs on the CIFAR-10 + "
            "frozen-probe protocol (SSR_PLAN §6) — lands with the S2 probe path.")

    os.makedirs(args.output, exist_ok=True)
    t_wall = time.time()
    print("=" * 72)
    print(f"SSR — model={args.model}, corruption={args.corruption}, arms={arms}, "
          f"attach={args.attach}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    rows = read_profile_csv(args.profile_csv)
    if not rows:
        raise ValueError(f"empty profile CSV: {args.profile_csv}")
    scope_rows = map_scope_rows(rows, args.attach)
    if not scope_rows:
        raise ValueError(f"--attach {args.attach!r} matches no rows in "
                         f"{args.profile_csv} (profiled modules: "
                         f"{[r['module'] for r in rows[:8]]} ...)")
    if args.k is not None:
        selected_rows = select_topk(scope_rows, args.k)
        k_desc = f"top-{args.k}/{len(scope_rows)} @{args.attach}"
    else:
        selected_rows = select_topk_fraction(scope_rows, args.k_fraction)
        k_desc = (f"top-{len(selected_rows)}/{len(scope_rows)} "
                  f"({args.k_fraction:.0%}) @{args.attach}")
    selected_names = [r["module"] for r in selected_rows]
    print(f"Drift selection ({k_desc} by cka_unbiased_drop): {selected_names}")

    n_pool = args.n_est
    n_total = n_pool + args.n_images
    if args.cache_dir:
        images, labels = rdp_main.load_imagenet_cached(
            n_total, args.seed, args.data_root, args.cache_dir)
    else:
        images, labels = load_imagenet_subset(
            n_images=n_total, seed=args.seed, data_root=args.data_root)
    pool_imgs, pool_lbls = images[:n_pool], labels[:n_pool]
    eval_imgs, eval_lbls = images[n_pool:], labels[n_pool:]
    print(f"pool={n_pool} (calibration + estimation), eval={len(eval_imgs)} "
          f"(disjoint, SSR_PLAN §6)")

    model, model_desc = rdp.load_model(args.model, DEVICE)
    print(f"Model: {model_desc}")
    modules = select_modules_for_hooks(model, args.max_depth, args.attach)
    dummy = [pool_imgs[0]]
    modules, axes = probe_channel_axes(model, modules, dummy, DEVICE,
                                       args.vit_token)
    if not modules:
        raise ValueError(f"--attach {args.attach!r} matched no firing modules "
                         f"on {args.model}")
    missing = [n for n in selected_names if n not in modules]
    if missing:
        raise ValueError(f"selected modules not hookable on this model: {missing}")
    print(f"{len(modules)} modules fire; channel axes: "
          f"{ {n: axes[n] for n in selected_names} }")

    t0 = time.time()
    clean_stats = collect_stats(model, pool_imgs, modules, axes, args.batch_size,
                                DEVICE, args.vit_token)
    print(f"[+{time.time() - t0:.0f}s] clean calibration moments on {n_pool} images")

    calib_d = []
    B = max(1, args.est_batches)
    step = max(1, n_pool // B)
    for i in range(0, n_pool, step):
        batch = pool_imgs[i:i + step]
        st = collect_stats(model, batch, modules, axes, args.batch_size,
                           DEVICE, args.vit_token)
        calib_d.append(aggregate_distances(
            list(shift_distances(clean_stats, st, list(clean_stats)).values())))
    tau = fit_gate(calib_d, quantile=args.gate_quantile, margin=args.gate_margin)
    print(f"Gate tau={tau:.6f} (clean distances max={max(calib_d):.6f})")

    with torch.no_grad():
        sample = eval_imgs[:args.batch_size]
        tensors = torch.stack([rdp.PREPROCESS(i) for i in sample]).to(DEVICE)
        out = model(tensors)
        id_logits_frozen = (out[0] if isinstance(out, (tuple, list)) else out).clone()

    aqr_source = None
    if "aqr" in arms:
        aqr_acts = collect_activations(model, pool_imgs, modules, axes,
                                       args.batch_size, DEVICE, args.vit_token)
        aqr_source = {n: fit_channel_quantiles(a, channel_axis=1)
                      for n, a in aqr_acts.items() if n in selected_names}
        print(f"AQR source percentiles fitted on {len(aqr_source)} modules")

    fn = get_corruption(args.corruption)
    results, identity_report = [], None

    for sev in severities:
        est_imgs = corrupt_pool(pool_imgs, fn, sev, 1000 + sev)
        eval_deg = corrupt_pool(eval_imgs, fn, sev, 1000 + sev)

        est_stats = collect_stats(model, est_imgs, modules, axes, args.batch_size,
                                  DEVICE, args.vit_token)
        d = shift_distances(clean_stats, est_stats, list(clean_stats))
        agg = aggregate_distances(list(d.values()))
        g_open = gate_open(agg, tau)
        transforms = {}
        for n in selected_names:
            a, b = moment_affine(clean_stats[n], est_stats[n], clip=args.clip)
            transforms[n] = affine_transform(a, b, args.clip)

        if sev == severities[0] and identity_report is None:
            # C2 (SSR_PLAN §5): on CLEAN inputs the gate must close and the
            # pipeline must be bit-identical to frozen. Two sub-checks:
            # (a) gate closes on a clean estimation pass (no false positive);
            # (b) with the gate closed, attaching the affine hooks (disarmed)
            #     must not perturb a single logit — the bypass contract.
            clean_est = collect_stats(model, pool_imgs, modules, axes,
                                      args.batch_size, DEVICE, args.vit_token)
            d_clean = shift_distances(clean_stats, clean_est, list(clean_stats))
            agg_clean = aggregate_distances(list(d_clean.values()))
            gate_closed_on_clean = not gate_open(agg_clean, tau)
            hooks = RecalibrationHooks(transforms, axes).attach(modules)
            try:
                hooks.armed = False  # gate closed -> bypass (recalibrate contract)
                with torch.no_grad():
                    out2 = model(tensors)
                    out2 = out2[0] if isinstance(out2, (tuple, list)) else out2
                maxdiff = float((out2 - id_logits_frozen).abs().max())
            finally:
                hooks.remove()
            identity_report = {
                "clean_shift_distance": agg_clean,
                "gate_closed_on_clean": bool(gate_closed_on_clean),
                "disarmed_hooks_max_abs_logit_diff": maxdiff,
                "bit_identical": bool(gate_closed_on_clean and maxdiff == 0.0),
            }
            print(f"Identity check (clean inputs): d={agg_clean:.6f} <= tau={tau:.6f} "
                  f"-> gate {'closed' if gate_closed_on_clean else 'OPEN'}; "
                  f"disarmed-hook max|dlogit|={maxdiff} -> "
                  f"{'BIT-IDENTICAL' if identity_report['bit_identical'] else 'BROKEN'}")

        for arm in arms:
            t_arm = time.time()
            top1 = None
            arm_transforms = None
            restore = None
            used_gate = None
            if arm == "frozen":
                top1 = evaluate(model, eval_deg, eval_lbls, args.batch_size, DEVICE,
                                args.vit_token)
            elif arm == "ssr_topk":
                used_gate = g_open
                if g_open:
                    arm_transforms = transforms
                top1 = run_eval_with(model, eval_deg, eval_lbls, modules, axes,
                                     args.batch_size, args.vit_token, arm_transforms)
            elif arm == "ssr_all":
                used_gate = g_open
                all_t = {}
                for n in modules:
                    if n in clean_stats and n in est_stats:
                        a, b = moment_affine(clean_stats[n], est_stats[n],
                                             clip=args.clip)
                        all_t[n] = affine_transform(a, b, args.clip)
                if g_open:
                    arm_transforms = all_t
                top1 = run_eval_with(model, eval_deg, eval_lbls, modules, axes,
                                     args.batch_size, args.vit_token, arm_transforms)
            elif arm == "ssr_gate_off":
                top1 = run_eval_with(model, eval_deg, eval_lbls, modules, axes,
                                     args.batch_size, args.vit_token, transforms)
            elif arm == "bn_stats":
                scope = selected_names if args.bn_scope == "topk" else None
                restore = bn_stat_replace(model, est_imgs, args.batch_size, DEVICE,
                                          scope_modules=scope)
                try:
                    top1 = evaluate(model, eval_deg, eval_lbls, args.batch_size,
                                    DEVICE, args.vit_token)
                finally:
                    restore()
            elif arm == "tent":
                kind = "bn" if any(isinstance(m, torch.nn.BatchNorm2d)
                                   for m in model.modules()) else "ln"
                ok = tent_adapt(model, est_imgs, args.batch_size, DEVICE, kind,
                                args.tent_lr, args.tent_steps)
                if not ok:
                    print(f"  WARNING: tent skipped ({kind}-affine params absent)")
                else:
                    top1 = evaluate(model, eval_deg, eval_lbls, args.batch_size,
                                    DEVICE, args.vit_token)
            elif arm == "aqr":
                tgt = collect_activations(model, est_imgs,
                                          {n: modules[n] for n in selected_names},
                                          axes, args.batch_size, DEVICE,
                                          args.vit_token)
                aqr_t = {}
                for n in selected_names:
                    if n in aqr_source and n in tgt:
                        aqr_t[n] = aqr_transform(
                            aqr_source[n], fit_channel_quantiles(tgt[n], channel_axis=1))
                top1 = run_eval_with(model, eval_deg, eval_lbls, modules, axes,
                                     args.batch_size, args.vit_token, aqr_t)
            elif arm == "clahe":
                from src.corruptions import clahe_enhance
                enhanced = [clahe_enhance(img) for img in eval_deg]
                top1 = evaluate(model, enhanced, eval_lbls, args.batch_size,
                                DEVICE, args.vit_token)
            results.append({
                "model": args.model, "corruption": args.corruption,
                "severity": sev, "arm": arm, "top1": top1,
                "shift_distance": agg, "gate_open": g_open, "tau": tau,
                "modules_touched": (len(selected_names) if arm_transforms else 0),
                "arm_wall_s": time.time() - t_arm,
            })
            print(f"  sev{sev} {arm:<12} top1={top1:.4f} "
                  f"(d={agg:.3f}, gate={'open' if g_open else 'closed'})")

    frozen = {r["severity"]: r["top1"] for r in results if r["arm"] == "frozen"}
    for r in results:
        r["delta_vs_frozen"] = (r["top1"] - frozen[r["severity"]]
                                if r["severity"] in frozen else None)

    csv_path = os.path.join(args.output, "ssr_results.csv")
    fields = ["model", "corruption", "severity", "arm", "top1", "delta_vs_frozen",
              "shift_distance", "gate_open", "tau", "modules_touched", "arm_wall_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in results)
    print(f"  Saved {csv_path}")

    meta = {
        "model_desc": model_desc, "corruption": args.corruption,
        "attach": args.attach,
        "profile_csv": args.profile_csv, "selection": k_desc,
        "selected_modules": selected_names, "n_pool": n_pool,
        "n_eval": len(eval_imgs), "seed": args.seed,
        "gate": {"tau": tau, "clean_distances": calib_d,
                 "quantile": args.gate_quantile, "margin": args.gate_margin},
        "identity": identity_report,
        "per_condition_shift": {
            f"{args.corruption}@sev{sev}": float(
                r["shift_distance"]) for r in results if r["arm"] == "frozen"},
        "arms": arms, "device": str(DEVICE), "torch": torch.__version__,
        "wall_s": time.time() - t_wall,
    }
    meta_path = os.path.join(args.output, "ssr_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"  Saved {meta_path}")
    print(f"SSR_DONE arms={arms} wall={time.time() - t_wall:.0f}s")


def run_eval_with(model, images, labels, modules, axes, batch_size,
                  token_mode, transforms):
    """Eval with optional recalibration hooks armed (present iff non-empty)."""
    hooks = None
    if transforms:
        hooks = RecalibrationHooks(transforms, axes).attach(modules)
        hooks.armed = True
    try:
        return evaluate(model, images, labels, batch_size, DEVICE, token_mode)
    finally:
        if hooks is not None:
            hooks.remove()


if __name__ == "__main__":
    main()

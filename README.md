# DINOv2 Low-Light Robustness Study

**How badly does DINOv2 break in the dark, where does it break, why — and can we fix it by training 1% of the network?**

This project stress-tests Facebook's **DINOv2 ViT-S/14** (frozen, self-supervised backbone) against progressive low-light degradation on **CIFAR-10**, localizes the failure inside the network, identifies its mechanism, and remediates it with LoRA adapters.

---

## The Three-Phase Pipeline

```
Phase 1: MEASURE          Phase 2: LOCALIZE & EXPLAIN        Phase 3: REMEDIATE
run_notebook1.py          run_notebook2.py                   run_lora_simple_colab.py
1000 imgs, 6 severities   500 imgs + layer hooks + FFT       LoRA rank-8 on attention
linear probe accuracy     CKA drift, frequency ablation,     70% dark / 30% clean diet
+ embedding drift         bootstrap CIs                      10 epochs, T4 GPU
```

**Corruption model:** `low_light()` scales pixel intensity down and adds Gaussian noise — severity 0 = clean, severity 5 = near-black + heavy noise (ImageNet-C style).

---

---

## Deliverables & how to run them

| Deliverable | Location | Run / Regenerate |
|---|---|---|
| Presentation app (live, interactive) | `apps/presentation.py` | `streamlit run apps/presentation.py` |
| Internal smoke check (deterministic, no server) | `apps/presentation.py` | `PRESENTATION_SMOKE=1 python3 apps/presentation.py` |
| Single-section static render (PNG) | `apps/presentation.py` | `PRESENTATION_STATIC=1 PRESENTATION_SECTION=<name> python3 apps/presentation.py` |

`apps/presentation.py` is the single presentation layer for the study. It is **read-only with respect
to results**: it reads committed artifacts (`output/`, `colab_results/`) and rebuilds charts from them
(no new experiments). It has **14 sections with sidebar navigation**: home (story), methods, collapse, localize, mechanism, proxy gate, fix, 9-arm grid, families, readout, ExDark, ViT-B, bugs, figure gallery. The smoke check renders every one of those sections headlessly and exits non-zero if any
fails — it erases any doubt about whether the app actually renders every section from committed
data, and it can be wired into CI.


Each phase answers one question and hands its finding to the next:

| Phase | Question | Answer | Hands to next phase |
|-------|----------|--------|---------------------|
| **1 — Measure** | *How bad is it?* | Accuracy collapses **0.91 → 0.09**; embeddings drift to near-orthogonality (cos 1.00 → 0.16) | The failure is real, severe, and representational — *but where?* |
| **2 — Localize & explain** | *Where and why?* | CKA pins the drift on **late attention blocks** (max drop 0.81 at block 10 vs 0.58 at block 0, the earliest); frequency ablation shows the model runs on **low-frequency luminance** — exactly what darkness removes | The failure has an address (late attention) and a mechanism (lost low-freq structure) — *so fix that, precisely* |
| **3 — Remediate** | *Can it be fixed cheaply?* | LoRA on those attention layers (0.99% of params, 70/30 dark/clean diet) restores **mean 0.59 → 0.82**, **worst-case 5.1×**, clean accuracy unchanged-or-better | Confirms the causal story: fix the localized shift, recover the robustness |

All phases on one axis — accuracy per severity (Phase 3 fixed-augmentation run; the original column matches Phase 1 within eval-noise jitter):

| Severity | Original DINOv2 | Embedding drift (cos sim) | + LoRA | Gap recovered |
|----------|-----------------|---------------------------|--------|---------------|
| 0 (clean) | 0.913 | 1.000 | 0.953 | clean gets *better* |
| 1 | 0.903 | 0.946 | 0.953 | +0.050 |
| 2 | 0.860 | 0.810 | 0.937 | +0.077 |
| 3 | 0.603 | 0.589 | 0.883 | +0.280 |
| 4 | 0.213 | 0.309 | 0.807 | +0.593 |
| 5 (darkest) | 0.073 | 0.161 | 0.377 | +0.303 |

The drift column is the Phase 1/2 fingerprint: accuracy loss tracks embedding drift almost linearly. The LoRA column shows the recovery tracks the *same axis back up*. One phenomenon, measured, explained, and reversed.

---

## Results

### Phase 1 — DINOv2 is not robust to darkness

Linear probe (logistic regression on frozen embeddings), 1000 test images:

| Severity | Accuracy | Cosine sim to clean embeddings |
|----------|----------|-------------------------------|
| 0 (clean) | 0.913 | 1.000 |
| 1 | 0.900 | 0.946 |
| 2 | 0.860 | 0.810 |
| 3 | 0.603 | 0.589 |
| 4 | 0.220 | 0.309 |
| 5 (darkest) | 0.093 | 0.161 |

Accuracy falls **0.82 points** and embeddings drift to near-orthogonality (cos ~0.16) — near-flat through severity 2, collapsing steeply between severity 2 and 4.

![Phase 1 accuracy and drift](output/notebook1/dinov2_lowlight_results.png)

### Phase 2 — The failure is localized, systematic, and low-frequency

**Layer-wise CKA** — hooks on all 12 transformer blocks; CKA drop (clean → severity 5) by
layer, full matrix in `output/notebook2/cka_matrix.csv`:

| Layer | CKA drop |
|-------|----------|
| block 0 (earliest) | 0.58 |
| block 5 | 0.53 |
| **block 10** | **0.81** ← max drift |
| block 11 (final) | 0.78 |

→ Degradation is **not uniform**: late attention layers shift their representations far more than early ones.

**Frequency ablation** (low-pass vs. high-pass filtered inputs): low-pass-filtered images retain near-clean accuracy (~0.92 at severity 1) while high-pass destroys it (~0.59) — at every severity the low-frequency band dominates.

→ DINOv2 leans on **low-frequency luminance structure**, which is precisely what darkness removes.

![Layer-wise CKA](output/notebook2/layerwise_cka.png)

![Frequency test](output/notebook2/frequency_test.png)

### Phase 3 — LoRA on late attention layers recovers most of the loss

LoRA adapters (rank 8, α 16) on QKV/output projections, **221K trainable params = 0.99%** of the network, trained 10 epochs on 5000 images (70% darkened / 30% clean). Free-tier Colab T4, ~10 minutes of training. This is the **run of record**: per-sample seeded augmentation RNG and matched-noise evaluation (see `docs/METHODS.md` §7.3, §9.6–9.7); artifacts in `colab_results/lora_run_fixed/`.

| Severity | Original | LoRA | Δ |
|----------|----------|------|-----|
| 0 (clean) | 0.913 | 0.953 | +0.040 |
| 1 | 0.903 | 0.953 | +0.050 |
| 2 | 0.860 | 0.937 | +0.077 |
| 3 | 0.603 | 0.883 | **+0.280** |
| 4 | 0.213 | 0.807 | **+0.593** |
| 5 (darkest) | 0.073 | 0.377 | **+0.303** |
| **Mean** | **0.594** | **0.818** | **+0.224** |

**Worst-case accuracy improved 5.1×** (0.073 → 0.377), the near-failure regime (severity 4) recovered to 0.81, and clean accuracy *rose* — no catastrophic forgetting from the mixed diet.

![Original vs LoRA accuracy](colab_results/lora_run_fixed/lora_vs_orig_accuracy.png)

![LoRA training curves](colab_results/lora_run_fixed/lora_training_curves.png)

**Phase 3 v2 — the full comparison grid.** The Phase 0 refactor of `run_lora_simple_colab.py` turned each extension into a flag, enabling the three arms the original single-run design lacked: multi-seed error bars, CKA-guided rank allocation, and the full fine-tuning baseline. Artifacts: `colab_results/sessionD/`; full discussion in `docs/RESEARCH.md` §6.1–6.2.

| Arm | Trainable params | Mean acc ± SD (3 seeds) | Sev 5 mean | Clean mean |
|-----|-----------------|--------------------|-----------------|---------------|
| Original (frozen) | 0 | 0.598 | 0.083 | 0.913 |
| Uniform LoRA r8 | 221,184 (0.99%) | 0.827 ± 0.012 (0.815–0.839) | 0.390 | 0.972 |
| Late-only (blocks 9–11, r8) | 55,296 (0.25%) | 0.686 | 0.160 | 0.937 |
| **Drift-weighted ranks (∝ CKA drop)** | **172,800 (0.78%)** | **0.825 ± 0.016 (0.811–0.842)** | 0.371 | 0.973 |
| Full fine-tuning | 22,056,576 (100%) | 0.831 ± 0.019 (0.810–0.846) | 0.436 | 0.960 |

**Read:** CKA-guided ranks match uniform LoRA and full fine-tuning at **0.78%** of the trainable parameters — now with 3-seed error bars on every headline arm, all ranges fully overlapping; full FT buys no extra mean robustness and posts the *worst* clean accuracy (forgetting cost, replicating across seeds); late-only shows the drift profile needs a total-budget floor. Per-arm SDs (±1.2–1.9 points) set the resolution of any comparison — parity is claimed, superiority is not. Per-severity cells for all nine arms: `docs/RESEARCH.md` §6.1; 3-seed table: `output/seed_replication/seed_analysis.csv`.

**Corruption-family expansion (blur / JPEG / contrast).** The failure and the fix are not low-light-specific. Phase-1 fragility: severity-5 accuracy 0.143 (blur), 0.273 (JPEG), 0.840 (contrast) vs 0.087 (low-light). CKA drift profiles are late-concentrated for every *frequency-destroying* corruption (low_light, blur, JPEG — peaks at blocks 8–10) and ~flat for contrast (frequency-preserving, max drop 0.30). LoRA arms on free-tier GPU (Kaggle T4, `colab_results/kaggle_sessionF/`) and recovered from retained adapters (`run_blur_adapter_eval.py`): **JPEG + drift-weighted ranks 0.586 → 0.779 (+19.3 pts, the study's biggest win)**; **blur drift 0.526 → 0.703 (+17.7) vs uniform +17.2** (drift edges worst-case, sev-5 0.303 vs 0.270 — the low-light signature repeating); contrast uniform +7.5 vs drift +6.9 — the allocation rule gracefully matches uniform exactly where its diagnostic says there is nothing to reallocate. Gains track fragility across the grid: low_light +22.4 > JPEG +19.3 ≈ blur +17.7 > contrast +7. Full story: `docs/RESEARCH.md` §5.1, §6.3; protocol: `docs/METHODS.md` §7.6, §11.

**Readout-staleness decomposition (`output/readout_repair/`).** How much of the collapse is a stale linear head vs broken features? A severity-adapted probe (retrained per severity on the same train fold) recovers **+8.7 pts at sev-3, +25.3 at sev-4, +30.0 at sev-5** (n = 1,000, paired-permutation p ≤ 0.0032) — yet still plateaus at 0.377 vs 0.913 clean, and its predictions collapse onto one class as darkness deepens (top1_share 0.14 → 0.73, dominant *frog*). Readout staleness is real; feature drift remains the binding constraint — which is why LoRA (features) works and why the fixed-readout LoRA result already sits at the adapted-readout ceiling. Upstream's ViT-B pilot (+22.5 pp, n = 120) confirmed at proper power: `docs/RESEARCH.md` §6.4.

**Robustness checks.** The drift-allocation rule survives switching to the unbiased CKA estimator (blocks 9–11 keep max rank; 3 of 12 block ranks shift by one — `docs/METHODS.md` §4), and the CKA pipeline is deterministic across CPU and T4 to ~1e-6, so CPU- and GPU-produced artifacts are directly comparable. The drift *profile* itself replicates on a third architecture: DINOv2 ViT-B/14 (86M params) profiled across low_light + jpeg, severities 1–5, n = 1,000 (`output/vitb_profile/`, protocol `docs/METHODS.md` §7.8) is again late-heavy (block drops 0.80–0.87 at blocks 9–11, low-light sev 5) — and adds a sublayer finding the ViT-S whole-block CKA could not see: **MLP sublayers drift more than attention in all 12 blocks under both corruptions**. The Track-1 proxy-gate NO-GO (cheap amplitude proxies fail to rank CKA drift; jpeg ρ ≈ 0.88 but low_light ≤ 0.69) also reproduces on ViT-B — full CKA stays the profiler (`docs/RESEARCH.md` §5.1, `docs/DPA_DESIGN.md` §1).

**Sim-to-real (ExDark, 7,363 real low-light photographs).** Frozen DINOv2 probes at **0.725** with a *flat* darkness-response curve (darkest quintile 0.713 vs brightest 0.704); CLAHE buys **+0.001** — real darkness within ExDark's range does not reproduce the synthetic collapse. Synthetic-dark adapters (drift arm, trained only on CIFAR darkness) lift real-dark accuracy to **0.743 (+1.9 pts)**; the uniform arm transfers **0.741 (+1.6 pts)** — indistinguishable, matching the synthetic-grid parity. Genuine transfer, but only ~9% of the +21-point gain the same adapters buy on the synthetic severity axis. The study quantifies the sim-to-real gap rather than assuming it away.

---

## The Story in One Paragraph

DINOv2's low-light failure is **not** diffuse noise sensitivity. It is a systematic representational shift concentrated in the **late attention layers** (CKA drop 0.81 at block 10 vs 0.58 at block 0, the earliest block), driven by the loss of **low-frequency luminance structure** — the exact signal the model depends on most. Because the failure is localized, a **targeted 0.99%-parameter intervention** (LoRA on attention, trained on a dark/clean mix) recovers +0.22 mean accuracy and 5.1× worst-case robustness at a training cost of ~10 GPU-minutes — and letting the CKA diagnostic allocate the adaptation budget (drift-weighted ranks) matches both uniform LoRA and full fine-tuning at 0.78% of the parameters. On real darkness the story sharpens: ExDark accuracy is flat across luminance and enhancement buys nothing, yet synthetic-dark adapters still transfer +1.9 points — synthetic extreme darkening and real low light are different regimes, and the gap is now measured, not assumed.

---

## Running It

### Local (CPU, ~10–25 min per analysis notebook)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_notebook1.py    # Phase 1 → output/notebook1/
python run_notebook2.py    # Phase 2 → output/notebook2/
```

### Colab GPU (LoRA, ~15 min on free-tier T4)

Paste or upload `run_lora_simple_colab.py` into a Colab notebook and run — it installs its own deps and writes to `/content/output/`. Via the Colab CLI:

```bash
colab new -s lora_run --gpu T4
colab upload -s lora_run run_lora_simple_colab.py /content/run_lora_simple_colab.py
colab exec -s lora_run -f launch_lora.py    # nohup-detached launcher
```

### Kaggle GPU (corruption-family grid, free tier)

Kaggle is now the primary GPU channel (persistent kernels, 30 GPU-h/week). Build the self-contained kernel (all runtime files embedded — script kernels ship only one file), push, poll, pull:

```bash
.venv/bin/python3 scripts/build_kaggle_kernel.py        # → /tmp/kaggle_kernel/
cd /tmp/kaggle_kernel && kaggle kernels push -p .       # enable_gpu: true + accelerator nvidiaTeslaT4
kaggle kernels status arindamtripathi/corrgrid-tail
kaggle kernels output arindamtripathi/corrgrid-tail -p /tmp/kg_out
```

The kernel hard-aborts without CUDA, mounts CIFAR-10 from a private Kaggle dataset, and runs the `grid_tail.sh` arm chain. Full protocol: `docs/METHODS.md` §11.

---

## Repository Layout

| Path | Role |
|------|------|
| `utils.py` | Shared library: corruption fns, DINOv2 loading, hook extraction, linear probe, CKA, bootstrap CI, plotting |
| `run_notebook1.py` / `run_notebook1_colab.py` | Phase 1 runner (local / Colab) |
| `run_notebook2.py` / `run_notebook2_colab.py` | Phase 2 runner: CKA, frequency ablation, bootstrap CIs (local / Colab) |
| `run_lora_simple_colab.py` | Phase 3: manual LoRA (no PEFT dependency) — parameterized (rank/layers/seed/corruption/model); produced the results above |
| `run_exdark_baseline.py` | ExDark real-dark suite: 12-class probe, darkness-response curve, CLAHE control, adapter transfer (streaming, feature cache) |
| `run_readout_repair.py` | Readout-staleness decomposition: fixed vs severity-adapted probe, paired permutation tests, collapse metrics |
| `cka_unbiased_recompute.py` | Unbiased-CKA (Kornblith App. B) robustness recompute of the drift-allocation profile |
| `stats_tools.py` | Torch-free statistics module: unbiased CKA, Wilson CI, paired/curve permutation tests, BH-FDR, participation ratio, readout-collapse metrics |
| `run_drift_proxy.py` | Track-1 drift-proxy harness: per-module forward-only proxies vs CKA agreement, pre-registered go/no-go gate (`output/drift_proxy/`) |
| `run_drift_profile.py` | Severity×corruption drift-profile sweep over one clean pass (Tracks 2+3); produced the ViT-B atlas `output/vitb_profile/` |
| `scripts/build_kaggle_kernel.py` | Builds the self-contained Kaggle kernel (base64-embedded package files + CIFAR mount + GPU guard); regenerates the v6 kernel byte-identically |
| `scripts/kaggle/grid_tail.sh` | Corruption-arm chain executed inside the Kaggle kernel |
| `run_lora_finetune_colab.py` | Phase 3 alternative using the PEFT library |
| `dinov2_*.ipynb` | Original notebook forms of all three phases |
| `output/notebook1/`, `output/notebook2/` | Local results: CSVs, accuracy/drift plots, CKA heatmap, bootstrap CI, frequency test |
| `colab_results/` | Colab-run artifacts (LoRA plots + log, earlier Phase 1/2 runs, Phase 3 v2 grid in `sessionD/`, corruption-family sessions E + F) |
| `colab_results/kaggle_sessionF/` | Kaggle T4 corruption-grid artifacts: jpeg/contrast drift + uniform arms, nb2_contrast Phase-2 run, full kernel + per-arm logs |
| `colab_gpu_bench.py`, `colab_probe.py` | Colab CLI probes: auth/runtime check + T4 throughput benchmark |
| `docs/RESEARCH.md` | **Full research narrative**: aim, hypotheses, methodology rationale, findings, bugs-as-lessons, limitations, future work |
| `docs/METHODS.md` | Formal methods appendix: corruption parameter tables, CKA math, seed registry, environment versions |
| `docs/DPA_DESIGN.md` | Design doc for Drift-Profiled Adaptation (DPA) + Selective Statistic Recalibration (SSR): gate outcome, allocation taxonomy, prior-art search log |

## Reproducibility Notes

- Backbone: `torch.hub` DINOv2 ViT-S/14, frozen; input 224×224, patch 14
- Phase 1/2 linear probes: stratified 70/30 train/test split (seed 42) on clean embeddings, evaluated per severity
- CKA: linear CKA with `sqrt` normalization (`CKA = HSIC(X,Y) / sqrt(HSIC(X,X)·HSIC(Y,Y))`), unbiased by clean-vs-degraded sample pairing
- Bootstrap: 1000 resamples, per-severity independent RNG seeds
- LoRA: AdamW with deduplicated parameter groups (head params excluded from the adapter group), lr as in script, batch 64, 10 epochs
- ExDark suite: stratified 70/30 split (seed 42) identical across all passes; adapters rebuilt from the rank config embedded in `lora_adapters.pt`

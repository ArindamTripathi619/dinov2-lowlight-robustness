# Track 4 Plan — Selective Statistic Recalibration (SSR)

*Created 2026-10-13. Execution plan and brainstorm for ROADMAP Track 4. Method
definitions and novelty statements live in `DPA_DESIGN.md` §4 — this plan cites
them and never contradicts them; if a search or run narrows a claim, §4 is
edited first and this plan follows.*

## 0. Decisions locked (2026-10-13, made with the user)

1. **Plan home:** this file. `DPA_DESIGN.md` §4 stays the claim/novelty source;
   `ROADMAP.md` Track 4 stays the budget/dependency source.
2. **Start now:** the CPU core (`src/ssr.py` + CI tests) lands in parallel with
   the Track 3 atlas kernel; GPU sessions wait for harvested profiles anyway
   (selection input = Track 3 output).
3. **TENT baseline:** native on ResNet-50; on the five LayerNorm models run a
   clearly-labeled **LN-affine translation** (entropy-minimize the norm's affine
   params — the mechanism TENT uses, applied to the norm the model actually
   has), so every baseline spans all 6 models.

## 1. What SSR must prove (pre-registered claims)

- **C1 — Selectivity.** Recalibrating only the top-k drift modules matches or
  beats recalibrating everything while touching fewer modules.
- **C2 — Identity gate.** On clean inputs SSR is *bit-identical* to the frozen
  model — bypass by construction, not "close enough".
- **C3 — Label-free + backprop-free** (main variant): no labels, no gradients,
  forward-only estimation.
- **C4 — Cross-family.** Works on CNN and ViT-family models, including the
  five with no BatchNorm (§2).

**Honest-null pre-registration** (same discipline as the allocation-shape
nulls, RESEARCH §6.6): if SSR-all ≥ SSR-top-k, the selectivity claim fails and
the fallback frame is the DPA frame — *same result at lower cost* (modules
touched instead of params trained). We report that null rather than hunting
for a k where selection wins.

## 2. The structural issue: 5/6 models have no BatchNorm

Roster (Track 3): ResNet-50 (**BN**), ConvNeXt-T, ViT-S/16, Swin-T, DeiT-S,
DINOv2 ViT-B/14 (all **LayerNorm**). Consequences:

- Schneider-style norm-stat replacement (nearest neighbor, DPA_DESIGN §4) is
  **ResNet-50-only** — a baseline on one model, not the method.
- **Primary mechanism: per-channel affine moment matching at module outputs.**
  Store clean (μ_c, σ_c) per module once; from B unlabeled shifted batches
  estimate (μ_t, σ_t); apply `y' = a·y + b` with `a = clip(σ_c/σ_t)`,
  `b = μ_c − a·μ_t`. Forward-only, architecture-agnostic, no norm layer
  required — works on any module output (post-conv, post-LN, residual stream).
- On ResNet-50 run **both** variants (affine + true BN-stat replacement) for an
  apples-to-apples comparison against Schneider adapt-all under one protocol.
- **Attachment point is a design parameter for the smoke run:** LN normalizes
  per sample, so post-LN outputs may already look statistically tame; the
  corruption-driven channel moments likely live at pre-norm/residual-stream
  and attention/MLP outputs — exactly where the atlas profiles point. Settle on
  T4 smoke (S1), not on paper.

## 3. Prior-art gates — **DONE 2026-10-13**, verdicts in `DPA_DESIGN.md` §6

1. **Feature-statistic moment matching as TTA** — searched (AdaBN, TTN-line,
   DPTA, ESBN, neuron editing). **Closest neighbor: AQR (arXiv:2511.03148)** —
   source-percentile storage + per-channel quantile transform, BN/GN/LN-agnostic,
   backprop-free. It does *not* select by measured drift (static all/top-half
   heuristic) and has *no* clean gate — §4's narrow statement survives.
2. **Shift detection / when-to-adapt gating** — searched (confidence-gated
   adaptation, modality-selective TTA, TTA surveys). **No clean-calibrated
   distance gate with an exactly-identity guarantee found.**

**One narrowing recorded:** cross-norm-architecture applicability (C4) is no
longer novel per se — AQR already did it. C4 stays as a *demonstration
requirement*, not a claim; the claimable pair remains C1 (drift-ranked
selection) + C2 (identity gate).

**Two consequences for this plan:**
- **AQR joins the §6 arms list** — strongest published backprop-free TTA, and
  the reviewer-facing contrast: AQR App. D shows selection matters (top-half
  63.7 vs all-blocks 36.25 on ViT-B/ImageNet) but picks by heuristic; SSR picks
  by measured profile. Implement as per-channel piecewise-linear quantile map
  from clean-calibration percentiles (setup mirrors SSR's clean-moment storage).
- **moment (affine) vs quantile (AQR)** is now a mechanism ablation candidate.

**Refresh run 2026-10-14 (§10's "prior-art searches" executed in parallel with
the Track 3 v4 kernel):** GALA (AAAI 2025, layer selection by gradient alignment)
and ADAPT (NeurIPS 2025, backprop-free Gaussian alignment) screened; TENT
mechanism re-confirmed for §0.3's TENT-LN labeling; C2 gate re-checked in fresh
vocabulary. **No narrowing — DPA_DESIGN §4 holds; log in DPA_DESIGN §6
(third round).**

## 4. Selection signal

- β_m = severity-5 CKA drop straight from the Track 3 atlas CSVs — **one
  measurement, two consumers** (DPA allocation + SSR selection), the coherence
  claim of the framework (DPA_DESIGN §1.4).
- Selection is **per (model, corruption family)** (DPA_DESIGN §4).
- k policy: **k = top 25% of profiled modules primary**, plus a k-sweep
  {1, 3, 5, 10, all} — the sweep *is* the required top-k-vs-all ablation and
  doubles as the sensitivity curve.

## 5. The identity gate (C2)

- Shift distance `d` = RMS standardized moment distance (per-channel mean
  shift in units of clean σ + log-std ratio) aggregated over candidate modules,
  computed on a clean calibration set → threshold τ from clean distances
  (max or high quantile × margin).
- Inference: `d ≤ τ` → **bypass entirely** (byte-identical by construction:
  no module is touched); `d > τ` → apply recalibration to top-k modules only.
- Verification lives in CI (`tests/run_tests.py`): identity test, moment-math
  closed-form test, selection test.
- Gate-off ablation measures clean-side damage of always-on recalibration.

## 6. Evaluation protocol (parity rules carried over)

- **Data:** the same ImageNet-val subset as the atlas (folder names = labels,
  used *only* for scoring); estimation batches B=8×64 **disjoint** from eval
  images; matched-noise `rng 1000 + severity`, seed 42, severity ladder 1–5,
  low_light + jpeg.
- **Seeds:** 3 × estimation-batch draws (what is actually stochastic here) —
  matches the ROADMAP budget line's "3 seeds".
- **Metrics:** top-1 vs severity per arm; Δ over frozen; identity-check result;
  modules touched; estimation overhead (per-image cost must be zero).
- **DINOv2 special case:** no ImageNet head → SSR runs on the existing
  CIFAR-10 + frozen-probe protocol, selection from its existing profile —
  comparable to Tracks 1–2.

**Arms:** frozen · **SSR top-k (primary)** · SSR all · SSR gate-off ·
BN-stat replacement (ResNet-50, both scopes) · TENT (native) / TENT-LN
(translation, §0.3) · **AQR** (added 2026-10-13 after the §3 search — strongest
published backprop-free TTA) · CLAHE front-end
(`src/corruptions.clahe_enhance`, in-repo) · supervised aug-only FT as a
labeled upper-bound reference. ZeroDCE needs external weights — deferred;
CLAHE covers the enhancement line for session 1.

## 7. Compute plan (~5–8 GPU-h budget, 2 sessions)

Forward count: 6 models × 2 corruptions × 5 severities × ~9 arms × 2k images ≈
1.1 M forwards ≈ 3–4 h T4 + TENT's short backprop. Staged to de-risk:

### 7.1 S1 — harness + smoke — **DONE 2026-10-14**

ResNet-50 (BN arm) + ViT-S (LN arm); gate verification on real T4;
attachment-point check. Budget ~1 h (actual: 34 min, §8).

### 7.2 S2 — attachment-point probe (decision taken 2026-10-15)

The §2 question — mechanistic failure vs attachment-point artifact — is
answered cheap-first: probe *where* the repair attaches before writing off
moment matching. The full 6-model grid is deferred to **S3, probe-gated**
(~3–4 h, only run if the probe rescues the repair).

- **Protocol:** S1's exact protocol (seed 42, sev 1+5, 512 eval / 128 pool,
  top-25% drift selection, gate + identity check, real ImageNet-val) with
  only `--attach` varying; corruption = **low_light only** (S1's maximal-harm
  family and the §2 luminance family); arms = frozen · ssr_topk ·
  ssr_gate_off · tent (adaptation-only control, unaffected by attachment).
- **Scopes** (rows → k = top 25% of the scope's rows):

  | scope | ViT-S modules | ResNet-50 modules | ViT rows → k | RN rows → k |
  |---|---|---|---|---|
  | `profiled` | S1's atlas-hooked set (control; must reproduce S1 low_light) | same | 38 → 10 | 25 → 6 |
  | `sublayer` | `blocks.N.attn/mlp` outputs (pre-norm stream) | bottleneck outputs `layerN.M` | 24 → 6 | 16 → 4 |
  | `stream` | `blocks.N` outputs (no `blocks` container — its tensor *is* `blocks.11`'s) | stage outputs `layer1..4` (no `avgpool` — readout, not stream) | 12 → 3 | 4 → 1 |
  | `norm` | post-LN `blocks.N.norm1/norm2` (drift rank aliased: norm1 ← attn, norm2 ← mlp — the atlas does not hook norms) | — (no profiled norm rows to borrow) | 24 → 6 | — |

- **Budget:** 7 legs × ~6–8 min ≈ 1 h T4 (kernel `arindamtripathi/ssr-s2`).
- **Readout:** per-scope Δ vs the `profiled` control at sev 1+5. If the
  pre-norm scopes (`sublayer`/`stream`) still lose like `profiled` did, the
  mechanism is written off honestly and SSR ships as **gate + selection**
  (both validated in S1); if a pre-norm scope wins, the label-free repair
  claim is unblocked and S3 runs.

Kernel reuses the `scripts/build_kaggle_kernel.py` base64-packaging pattern;
new `scripts/kaggle/ssr.sh` (S1) and `scripts/kaggle/ssr_s2.sh` (S2).

## 8. Implementation pieces

- **`src/ssr.py` (landed with this plan)** — torch-free NumPy core: channel
  moments, moment-matching affine, shift distance, gate, drift-ranked top-k
  selector. Importable in CI (torch-optional convention, AGENTS environment
  notes).
- `run_ssr.py` — torch runner; CLI mirrors `run_drift_proxy.py`. **Landed
  2026-10-14** (after the Track 3 harvest, as planned): consumes the atlas
  severity-5 CSVs for drift-ranked top-k selection; arms frozen / ssr_topk /
  ssr_all / ssr_gate_off / bn_stats (ResNet-50, all|topk scopes) / tent (BN
  native, LN translation per §0.3) / aqr (moment-vs-quantile mechanism
  ablation) / clahe; per-run identity check (C2) prints gate-closed +
  bit-identical and lands in ssr_meta.json. CPU smoke verified on both
  roster paths (ResNet-50 BN/conv and ViT LN/token) — all 8 arms exit 0,
  identity BIT-IDENTICAL on both.
- **S1 T4 smoke DONE 2026-10-14** (kernel `arindamtripathi/ssr-s1`,
  ResNet-50 + ViT-S × low_light/jpeg × sev 1+5, real ImageNet-val, 34 min,
  all invocations rc=0, identity bit-identical on both families).
  **Verdict: C1–C4 all hold, but the moment-matching repair is net-harmful
  at every open gate** (top-k loses 1.6–52.3 pts vs frozen; all-scope loses
  up to 72); TENT never loses; AQR and BN-stat replacement also fail at
  sev≥5; one large positive = CLAHE on ViT-S low_light sev5 (+22.3).
  Full arms table + interpretation in RESEARCH §6.7; protocol in METHODS
  §7.11. The open question this leaves — mechanistic failure vs
  attachment-point artifact (§2) — was resolved as a decision on
  2026-10-15: run the §7.2 probe.
- **S2 probe harness LANDED 2026-10-15** (run pending): `--attach
  {profiled,stream,sublayer,norm}` on `run_ssr.py`, backed by torch-free
  `module_in_scope` / `map_scope_rows` in `src/ssr.py` — the scope owns both
  the hook-module set and the selection rows (so k = 25% counts scope rows;
  norm rows borrow their sublayer's drift rank; the ViT `blocks` container
  is excluded from `stream` because its output tensor *is* `blocks.11`'s —
  selecting both would apply the moment affine twice to one value).
  `ssr_meta.json` records `attach`; the results CSV schema is unchanged.
  Driver `scripts/kaggle/ssr_s2.sh` (7 legs → `output/ssr_s2/<slug>_<scope>_low_light/`),
  kernel variant `ssr_s2` (fresh slug `arindamtripathi/ssr-s2`; exit verifier
  derived from the driver's legs — the S1 naming-bug regression test
  mirrored). 47/47 tests; CPU smoke green on all 6 scope legs the driver
  uses (identity bit-identical on each; k/rows as §7.2).
- Profile consumer reuses the existing CSV schema
  (`layer,severity,module,cka,cka_unbiased,drop_from_clean,cka_unbiased_drop`)
  — no new format.

## 9. Scientific risk (stated upfront)

Track 1's NO-GO showed **amplitude statistics cannot *rank* structural drift**
(ROADMAP Track 1; DPA_DESIGN §1.2). SSR uses amplitude statistics to *repair*
drift — not a strict contradiction (ranking ≠ repairing), but it tempers
expectations: gains may be modest and most visible on jpeg/low_light where
channel moments actually shift. If gains are small, the defensible SSR
contribution is still real: selectivity + identity gate + cross-family
applicability — with honest numbers either way.

## 10. Sequence

```
Track 3 kernel ✓ (2026-10-14, drift-atlas-v4, fail=0) → harvest ✓ (output/profiles/)
        → atlas ✓ (output/atlas/) → committed (7561daf + 93b53ca)
        │
        ├── (done) src/ssr.py + tests → prior-art searches (3 rounds, §3)
        │
        └── run_ssr.py ✓ (consumes output/profiles/*.csv per §8)
                 → S1 smoke ✓ (2026-10-14, ssr-s1, T4, rc=0 all legs)
                 → RESEARCH verdict ✓ (§6.7: C1–C4 hold, repair harmful)
                 → S2 decision taken 2026-10-15: attachment-point probe first
                       (§7.2; full 6-model grid deferred to S3, probe-gated)
                 → S2 harness ✓ (2026-10-15: --attach scopes, ssr_s2.sh,
                       kernel ssr_s2, 47/47 tests, CPU smoke green)
                 → ssr-s2 T4 run (pending) → verdict → RESEARCH §6.8
                       (pre-norm scope wins → S3 grid; all scopes lose →
                        write off moment matching, ship gate + selection)
```

Selection inputs of record (§4, one measurement two consumers): per-model
severity-5 rows in `output/profiles/<model>/drift_profile_sev5_{low_light,jpeg}.csv`
— note the ImageNet legs key `layer` by profiled-module row (15–38 rows incl.
`blocks.k.attn/mlp` sublayers and CNN stages); `tools/build_atlas.py`'s
`normalize_depth`-style rank mapping handles cross-family depth alignment, and the
DINOv2 row is the first early-heavy profile in the roster (RESEARCH §5.2) — top-k
selection there will pick *early* modules, a built-in falsification test for C1.

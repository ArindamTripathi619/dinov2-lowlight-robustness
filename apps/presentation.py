"""Presentation app: interactive, story-first walkthrough of the low-light study.

Reads committed artifacts directly (no new experiments). 14 sections with real
sidebar navigation: story -> methods -> collapse -> localize -> mechanism ->
proxy gate -> fix -> 9-arm grid -> families -> readout -> ExDark -> ViT-B ->
bugs -> figure gallery.

Run:     streamlit run apps/presentation.py
Smoke:   PRESENTATION_STATIC=1 PRESENTATION_SECTION=<name>  (renders
         __static_<name>.png next to this script, exits 0)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT / "output"
COLAB = PROJECT / "colab_results"

st.set_page_config(page_title="Low-Light Study \u2014 Interactive Walkthrough",
                   page_icon="\U0001F52C", layout="wide")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

SECTION_ORDER = [
    "home", "methods", "collapse", "localize", "mechanism", "proxy", "fix",
    "grid", "families", "readout", "exdark", "vitb", "bugs", "gallery",
]

SECTION_LABELS = {
    "home": "\U0001F3E0 The story in 60 seconds",
    "methods": "\U0001F52C How we did it (methods)",
    "collapse": "\U0001F4A5 1 \u00b7 The collapse",
    "localize": "\U0001F4CD 2 \u00b7 Where it broke",
    "mechanism": "\U0001F39B 3 \u00b7 Why it broke",
    "proxy": "\U0001F6A7 4 \u00b7 The proxy gate \u2014 NO-GO",
    "fix": "\U0001FA79 5 \u00b7 The fix \u2014 LoRA",
    "grid": "\U0001F9EE 6 \u00b7 The 9-arm grid",
    "families": "\U0001F300 7 \u00b7 Corruption families",
    "readout": "\U0001F4E1 8 \u00b7 Readout staleness",
    "exdark": "\U0001F311 9 \u00b7 Real darkness (ExDark)",
    "vitb": "\U0001F52C 10 \u00b7 ViT-B/14 cross-architecture",
    "bugs": "\U0001F41B Bugs & lessons",
    "gallery": "\U0001F5BC Figure gallery",
}

SECTION_TITLES = {
    "home": "The story in 60 seconds",
    "methods": "How we did it \u2014 data, corruptions, metrics, training",
    "collapse": "Phase 1 \u2014 How bad is it?",
    "localize": "Phase 2 \u2014 Where does it break?",
    "mechanism": "Mechanism \u2014 low-frequency luminance",
    "proxy": "Track 1 \u2014 cheap proxy gate (NO-GO)",
    "fix": "Phase 3 \u2014 can we fix it cheaply?",
    "grid": "Phase 3 v2 \u2014 the 9-arm, 3-seed grid",
    "families": "Corruption families \u2014 does the fix generalize?",
    "readout": "Readout staleness \u2014 features, not the head",
    "exdark": "ExDark \u2014 real darkness",
    "vitb": "Phase 2.7 \u2014 ViT-B/14 cross-architecture",
    "bugs": "Appendix \u2014 bugs & lessons",
    "gallery": "Figure gallery \u2014 every committed figure",
}


# ---------------------------------------------------------------------------
# Data loading (all artifacts are committed; nothing here runs an experiment)
# ---------------------------------------------------------------------------

@st.cache_data
def load_notebook1():
    return pd.read_csv(OUTPUT / "notebook1" / "dinov2_lowlight_results.csv")

@st.cache_data
def load_cka_matrix():
    return pd.read_csv(OUTPUT / "notebook2" / "cka_matrix.csv")

@st.cache_data
def load_cka_unbiased():
    return pd.read_csv(OUTPUT / "notebook2" / "cka_matrix_unbiased.csv")

@st.cache_data
def load_seed_analysis():
    import csv as _csv
    p = OUTPUT / "seed_replication" / "seed_analysis.csv"
    phase_rows = []
    with open(p, encoding="utf-8") as f:
        for row in _csv.reader(f):
            if len(row) == 7:
                phase_rows.append(row)
    df = pd.DataFrame(phase_rows, columns=["arm", "group", "seed", "mean_acc", "sev5_acc", "clean_acc", "status"])
    for col in ["mean_acc", "sev5_acc", "clean_acc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_readout():
    return pd.read_csv(OUTPUT / "readout_repair" / "results.csv")

@st.cache_data
def load_readout_summary():
    with open(OUTPUT / "readout_repair" / "summary.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_exdark_summary(subdir="exdark_baseline"):
    return pd.read_csv(OUTPUT / subdir / "results.csv")

@st.cache_data
def load_exdark_darkness(subdir="exdark_baseline"):
    return pd.read_csv(OUTPUT / subdir / "darkness_curve.csv")

@st.cache_data
def load_exdark_class(subdir="exdark_baseline"):
    return pd.read_csv(OUTPUT / subdir / "class_accuracy.csv")

@st.cache_data
def load_bridge_drift():
    return pd.read_csv(OUTPUT / "blur_reeval" / "blur_drift" / "results.csv")

@st.cache_data
def load_bridge_uniform():
    return pd.read_csv(OUTPUT / "blur_reeval" / "blur_uniform" / "results.csv")

@st.cache_data
def load_vitb_profiles():
    """37 modules x 5 severities x 2 corruptions, incl. biased/unbiased CKA."""
    return pd.read_csv(OUTPUT / "vitb_profile" / "proxy_profiles.csv")

@st.cache_data
def load_vitb_agreement():
    with open(OUTPUT / "vitb_profile" / "agreement_report.json") as f:
        return json.load(f)

@st.cache_data
def load_drift_proxy_agreement():
    with open(OUTPUT / "drift_proxy" / "agreement_report.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plot_acc_drift(df, title, y2=None):
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    sev = df["severity"].astype(int)
    if "accuracy" in df.columns:
        ax1.plot(sev, df["accuracy"], marker="o", color="#1f77b4", linewidth=2.2, label="Accuracy")
        ax1.set_ylabel("Accuracy", color="#1f77b4")
    else:
        ax1.plot(sev, df["drop_from_clean"], marker="o", color="#1f77b4", linewidth=2.2, label="Drop from clean")
        ax1.set_ylabel("Drop from clean (1 \u2212 CKA)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    if y2 is not None:
        ax2.plot(sev, df[y2], marker="s", color="#d62728", linewidth=2.2, label="Cosine drift")
        ax2.set_ylabel("Cosine sim to clean", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylim(0, 1.02)
    ax1.set_xlabel("Darkness severity (0 = clean \u2192 5 = darkest)")
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)
    fig.tight_layout()
    return fig


def _plot_layerwise(df, sev=5):
    sev = int(sev)
    sub = df[df["severity"] == sev]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    blocks = sub["layer"].astype(int).values
    drops = sub["drop_from_clean"].values
    colors = ["#d62728" if b >= 9 else "#1f77b4" for b in blocks]
    ax.bar([f"b{b}" for b in blocks], drops, color=colors, width=0.7)
    ax.set_xlabel("Block (0 = patch embed, 9\u201311 = late)")
    ax.set_ylabel("Drop from clean (1 \u2212 CKA)")
    ax.set_title(f"Drift by block \u2014 severity {sev} (drop {drops.min():.3f}\u2013{drops.max():.3f})")
    ax.axvspan(8.5, 11.5, color="#ff7f0e", alpha=0.08)
    ax.text(10, ax.get_ylim()[1] * 0.95, "late\nblocks", ha="center", va="top", fontsize=9, color="#b22222")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_heatmap(piv, title, cbar_label="Drop from clean (1 \u2212 CKA)", figsize=(7.6, 5.0)):
    """piv: DataFrame index=block/layer, columns=severity, values=drift drop."""
    fig, ax = plt.subplots(figsize=figsize)
    data = piv.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.0,
                   vmax=max(0.05, float(np.nanmax(data))))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"sev {c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index], fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 0.62 else "white")
    ax.set_xlabel("Severity")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_two_arm(df, col_a, col_b, title, label_a=None, label_b=None, xlabel="Severity"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sev = df["severity"].astype(int).values
    a = df[col_a].values
    b = df[col_b].values
    ax.plot(sev, a, marker="o", color="#1f77b4", linewidth=2.2, label=label_a or col_a)
    ax.plot(sev, b, marker="s", color="#2ca02c", linewidth=2.2, label=label_b or col_b)
    for i, s in enumerate(sev):
        ax.annotate(f"{a[i]:.3f}", (s, a[i]), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#1f77b4")
        ax.annotate(f"{b[i]:.3f}", (s, b[i]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#228B22")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(sev)
    ax.set_ylim(0, 1.02)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_readout(df):
    piv = df.pivot_table(index="severity", columns="arm", values="accuracy").reset_index()
    return _plot_two_arm(
        piv, "fixed", "adapted",
        title="Readout staleness: retraining the probe fixes severity 3\u20135, but plateaus at ~0.38",
        label_a="Arm A \u2014 probe fixed (stale readout)",
        label_b="Arm B \u2014 probe retrained per severity",
        xlabel="Low-light severity",
    )


def _plot_proxy(df, title):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    proxies = ["energy_drop", "mean_shift", "cov_shift", "cos_drop"]
    colors = {"energy_drop": "#1f77b4", "mean_shift": "#ff7f0e", "cov_shift": "#2ca02c", "cos_drop": "#d62728"}
    for p in proxies:
        sub = df[df["proxy"] == p]
        ax.plot(sub["corruption"], sub["rho"], marker="o", color=colors[p], label=p)
    ax.axhline(0.8, color="#9467bd", linestyle="--", linewidth=1.2, label="Gate threshold \u03c1 = 0.8")
    ax.set_xlabel("Corruption")
    ax.set_ylabel("Spearman \u03c1 (proxy \u2194 CKA)")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_seed_grid(df, metric="mean_acc"):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    arms = df.groupby("arm")[metric].agg(["mean", "std"]).reset_index()
    arms["lo"] = arms["mean"] - arms["std"]
    arms["hi"] = arms["mean"] + arms["std"]
    order = ["uniform", "drift", "fullft"]
    arms = arms.set_index("arm").reindex(order).reset_index()
    x = np.arange(len(arms))
    ax.bar(x, arms["mean"], yerr=[arms["mean"] - arms["lo"], arms["hi"] - arms["mean"]],
           capsize=6, color=["#1f77b4", "#2ca02c", "#ff7f0e"], width=0.55, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(arms["arm"])
    ax.set_ylabel("Mean accuracy (3 seeds)")
    ax.set_title("Drift-weighted matches uniform and full FT at 0.78% of parameters")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    for i, (_, r) in enumerate(arms.iterrows()):
        ax.text(i, r["mean"] + r["std"] + 0.015, f"{r['mean']:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_vitb_sublayers(df, sev=5, corr="low_light", value_col="cka_unbiased_drop"):
    """Grouped bars: attention vs MLP drift per block (ViT-B profile data)."""
    sub = df[(df["severity"] == sev) & (df["corruption"] == corr)]
    pat = sub[sub["module"].str.fullmatch(r"blocks\.\d+\.(attn|mlp)")].copy()
    pat["block"] = pat["module"].str.extract(r"blocks\.(\d+)").astype(int)
    pat["kind"] = pat["module"].str.extract(r"\.(attn|mlp)")
    piv = pat.pivot_table(index="block", columns="kind", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv["attn"], width=w, color="#1f77b4", label="attn sublayer")
    ax.bar(x + w / 2, piv["mlp"], width=w, color="#ff7f0e", label="mlp sublayer")
    ax.set_xticks(x)
    ax.set_xticklabels([f"b{i}" for i in piv.index])
    ax.set_xlabel("Block")
    ax.set_ylabel("Drift drop (1 \u2212 unbiased CKA)")
    ax.set_title(f"DINOv2 ViT-B/14 \u2014 {corr}, sev {sev}: MLP > attention in every block")
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_darkness_curve(df):
    d = df.sort_values("lum_mean")
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(d["lum_mean"], d["accuracy"], marker="o", color="#1f77b4", linewidth=2.2, label="Accuracy")
    ax1.set_xlabel("Mean luminance of bin (0\u2013255)")
    ax1.set_ylabel("Accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.bar(d["lum_mean"], d["n"], width=9, color="#bbbbbb", alpha=0.7, label="Images in bin")
    ax2.set_ylabel("Images in bin", color="#666666")
    ax2.tick_params(axis="y", labelcolor="#666666")
    ax1.set_title("Real darkness (ExDark): accuracy is flat across the luminance axis")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_class_acc(df):
    d = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = ["#d62728" if v < 0.6 else "#1f77b4" for v in d["accuracy"]]
    ax.barh(d["class"], d["accuracy"], color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Frozen-model accuracy")
    ax.set_title(f"ExDark per-class accuracy (n = {int(d['n'].sum())}, 12 classes)")
    ax.grid(True, axis="x", alpha=0.25)
    for i, (v, n) in enumerate(zip(d["accuracy"], d["n"])):
        ax.text(v + 0.015, i, f"{v:.2f} (n={n})", va="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def _img(rel, caption=None, width=None):
    """Show a repo-relative figure (str or Path); never crashes on a missing file."""
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT / p
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"missing figure: {rel}")


def _img_grid(pairs, columns=2):
    """pairs: list of (rel_path, caption). Renders a responsive figure grid."""
    rows = [pairs[i:i + columns] for i in range(0, len(pairs), columns)]
    for row in rows:
        cols = st.columns(columns)
        for c, (rel, cap) in zip(cols, row):
            with c:
                _img(rel, cap)


def _bullets(items):
    st.markdown("\n".join(f"- {it}" for it in items))


def _findings(items):
    st.subheader("Findings")
    st.markdown("\n".join(f"- {it}" for it in items))


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_home():
    st.subheader(SECTION_TITLES["home"])
    st.markdown("""
    **One paragraph, if you read nothing else**

    A frozen self-supervised vision model is brilliant on clean photos (91%) but
    collapses to ~9% under extreme darkness. It's not diffuse noise; it's a
    **localized representational shift in the late blocks**, driven by the loss
    of **low-frequency luminance** \u2014 exactly what darkness removes. Because the
    failure has an address, a **tiny 0.99%-parameter LoRA patch** (trained in
    ~10 GPU-minutes on a dark/clean mix) recovers most of it: mean accuracy
    **0.59 \u2192 0.82**, worst-case **5.1\u00d7**, and clean accuracy even **rises**.
    The same trick generalizes to JPEG, blur, and contrast. On real darkness
    (ExDark), the honest story is: synthetic extreme darkening is harsher than
    real low light, but the synthetic-dark adapters still transfer **+1.9 pts**
    \u2014 a gap now *measured*, not assumed.
    """)

    st.subheader("The hero figures")
    _img_grid([
        ("output/notebook1/dinov2_lowlight_results.png",
         "Phase 1 of record: accuracy + embedding drift vs darkness severity"),
        ("output/notebook1/severity_visual_check.png",
         "What severity 0\u20135 looks like on real CIFAR-10 images"),
    ])

    st.subheader("The numbers at a glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clean accuracy", "0.913", "4/10 on CIFAR-10")
    with col2:
        st.metric("Darkness (sev 5)", "0.093", "9.3 pts below clean")
    with col3:
        st.metric("LoRA mean", "0.818", "+0.224 mean across severities")
    with col4:
        st.metric("LoRA sev 5", "0.377", "5.1\u00d7 worst-case improvement")

    st.subheader("Headline results, question by question")
    headline = pd.DataFrame([
        {"Question": "How bad is extreme darkness?",
         "Answer": "0.913 \u2192 0.093 accuracy (cliff between sev 2\u20134; chance = 0.10)"},
        {"Question": "Where does the representation break?",
         "Answer": "Late blocks 9\u201311: CKA drop 0.81 vs 0.18 in the patch embed"},
        {"Question": "Why?",
         "Answer": "Darkness removes low-frequency luminance; low-pass keeps accuracy, high-pass destroys it"},
        {"Question": "Can a cheap proxy replace CKA?",
         "Answer": "NO-GO on 3 architectures (ViT-S, ViT-S-recheck, ViT-B) \u2014 full CKA stays the profiler"},
        {"Question": "Cheapest fix?",
         "Answer": "LoRA r8 on 24 modules = 0.99% of params, ~10 min on a free T4"},
        {"Question": "How much does it recover?",
         "Answer": "Mean 0.594 \u2192 0.818 (+22.4); worst-case 0.073 \u2192 0.377 (5.1\u00d7); clean accuracy rises"},
        {"Question": "Does drift-weighting beat uniform LoRA?",
         "Answer": "Parity: 0.825 / 0.827 / 0.831 (drift / uniform / full FT), 3-seed ranges overlap"},
        {"Question": "Does the fix generalize?",
         "Answer": "JPEG +19.3, blur +17.7, contrast +7.5 \u2014 gain tracks fragility"},
        {"Question": "Is it just the linear head going stale?",
         "Answer": "No \u2014 re-trained probe caps at 0.377 at sev 5; the majority of loss is in the features"},
        {"Question": "Real darkness (ExDark, 7,363 images)?",
         "Answer": "Frozen model 0.725, flat across luminance; synthetic-dark LoRA transfers +1.9 pts"},
        {"Question": "Cross-architecture?",
         "Answer": "ViT-B/14 confirms late-heavy drift; MLP drifts more than attention in all 12 blocks"},
    ])
    st.dataframe(headline, use_container_width=True, hide_index=True)

    st.subheader("Findings in plain language")
    st.markdown("""
    1. **The failure is a cliff, not a slope** \u2014 near-clean through severity 2, then a steep collapse.
    2. **It has an address** \u2014 the late transformer blocks move the most; the patch embed barely moves.
    3. **It has a mechanism** \u2014 the model runs on low-frequency luminance, which is exactly what darkness deletes.
    4. **There is no free lunch in profiling** \u2014 cheap amplitude proxies fail the correlation gate; forward-only CKA is cheap enough.
    5. **A 0.99%-parameter patch recovers most of it** \u2014 and doesn't trade away clean accuracy.
    6. **The fancy allocation doesn't matter (parity)** \u2014 drift-weighted ranks match uniform ranks and full fine-tuning.
    7. **The fix transfers** \u2014 to JPEG, blur, and contrast, with gains proportional to how fragile the family is.
    8. **The honest boundaries** \u2014 readout retraining caps out at 0.377; real-world transfer is +1.9, not +21.
    """)

    st.subheader("What we don't claim")
    _bullets([
        "We don't claim drift-weighting beats uniform (parity, 3-seed error bars show overlap).",
        "We don't claim the synthetic results carry to real darkness (we measure the gap instead).",
        "We don't claim the findings generalize to ImageNet-scale statistics (CIFAR-10 at 32\u00d732 upsampled).",
        "We don't claim a cheap proxy exists (the gate was NO-GO \u2014 full CKA is cheap enough, forward-only).",
    ])

    st.subheader("Walkthrough order")
    st.caption("New here? Start with **How we did it** \u2014 it tells the whole experimental story in plain words, before any numbers.")
    walk = [
        ("methods", "Data, corruption parameter tables, CKA formula, probe and LoRA configs."),
        ("collapse", "Accuracy cliff 0.91 \u2192 0.09 with embedding drift riding alongside."),
        ("localize", "Layer \u00d7 severity CKA matrix: late blocks 9\u201311 carry the drift."),
        ("mechanism", "Frequency controls: low-pass preserves, high-pass destroys."),
        ("proxy", "Track 1 gate: cheap proxies vs CKA \u2014 verdict NO-GO."),
        ("fix", "LoRA run of record: 0.99% params, +0.22 mean, 5.1\u00d7 worst-case."),
        ("grid", "9 arms \u00d7 3 seeds: drift = uniform = full FT, with per-arm training curves."),
        ("families", "Blur / JPEG / contrast: profile, fix, and per-family figures."),
        ("readout", "Probe-retrain control: how much was the head vs the features."),
        ("exdark", "7,363 real low-light images: the sim-to-real gap, measured."),
        ("vitb", "86M-parameter replication: late-heavy + MLP > attention in 12/12 blocks."),
        ("bugs", "Four bugs, each caught pre-trust, each with a reproducibility lesson."),
        ("gallery", "Every committed figure, grouped by phase."),
    ]
    for key, blurb in walk:
        st.markdown(f"**{SECTION_LABELS[key].split(' ', 1)[-1]}** \u2014 {blurb}")


def _render_methods():
    st.subheader(SECTION_TITLES["methods"])
    st.markdown("""
    The whole study is **one repeatable pipeline**, run end to end without ever training the big
    model. Here is that pipeline in plain words — ten decisions, why each one exists, and what
    would have gone wrong without it. The exact parameter tables sit underneath as reference cards.
    """)
    steps = [
        ("Freeze a strong model — and only ever question it", """
**What:** DINOv2 ViT-S/14 (22.3M parameters) from `torch.hub`, permanently in `eval()` +
`no_grad()`. CIFAR-10's 32×32 images are upsampled to the model's native 224×224,
ImageNet-normalized, and we read the pooled **CLS token** — a single 384-number summary of the
whole image — as "the representation".

**Why freeze it?** We are studying how an *existing* visual system degrades, not building a new
one. Every frozen weight is a weight that cannot drift behind our back: when accuracy falls, the
input is the cause — nothing we trained can be blamed.
"""),
        ("Build a dimmer switch instead of collecting dark photos", """
**What:** four synthetic degradations, each with six fixed severity steps (0 = untouched … 5 =
extreme): **low-light** (brightness ×1.00 → 0.15 plus a little Gaussian noise), **Gaussian blur**
(radius 0 → 5.5 px), **JPEG** (quality pristine → 8), **contrast** (fade toward gray, ×1.00 →
0.15) — plus two FFT frequency filters used only by the mechanism experiment.

**Why a fixed table instead of real photos?** Because anyone must be able to turn the same knob
and get the same image: the severity tables are literal code, so a rerun on any machine reproduces
the *identical* corrupted inputs. Severity 0 returning the pristine image is load-bearing — it is
the clean anchor every curve and every probe is measured against.
"""),
        ("Grade with the simplest classifier that exists (a linear probe)", """
**What:** a frozen model doesn't predict labels — it emits embeddings. So we train the simplest
classifier that exists, a **logistic regression** (384 → 10 classes), on **clean severity-0
embeddings only**, then evaluate that one probe at every severity (70/30 stratified split,
seed 42).

**Why clean-only training?** We're asking whether the features the model *already has* still carry
label information after degradation. Retraining on degraded data would measure *adaptation*
instead of *breakage* — a different and much less alarming question.

**Why the comparison is fair:** original and patched models receive the *identical* test indices,
so every before/after pair is same-images by construction.
"""),
        ("Measure drift with CKA — 'did the picture inside move?'", """
**What:** for the same image, clean vs degraded, **linear CKA** compares each block's activations
and returns a number in [0, 1]: 1.0 means the internal picture is arranged exactly as before, 0
means it has been reshuffled beyond recognition. Intuition: it compares the *structure* of the
representations (which samples sit near which other samples), not their raw values.

**Why CKA instead of a plain distance?** It is scale-invariant — a uniformly "louder"
representation isn't a *different* one — so what it measures is reorganization, which is exactly
the failure mode we care about. It needs **no labels**: two forward passes per condition, hooks on
each module, nothing else.

**Two honesty notes baked in:** (a) the original implementation forgot a square root — values were
squared, which kept every *ranking* intact but made the raw numbers meaningless; it was caught,
fixed, and everything reported here uses the corrected formula. (b) The biased estimator flatters
similarities, so the whole matrix was recomputed with the unbiased variant — the late-block
structure survives both.
"""),
        ("Put honest error bars on every accuracy (bootstrap)", """
**What:** every accuracy in this study comes from 1,000 graded images — one sample that could
wobble with luck. So per severity we **bootstrap**: resample the 1,000 per-image right/wrong answers
1,000 times with replacement and take the 2.5/97.5 percentiles → a 95% confidence band.

**Plain words:** re-roll the same exam 1,000 different ways and watch how much the class average
moves. A narrow band means the number is stable.

**The bug this caught:** the first version re-seeded every severity with the same seed, so all
severity levels resampled *identically* and the bands looked suspiciously similar. Fixed with
severity-indexed seeds — independence between the conditions we intend to compare.
"""),
        ("Localize the damage with per-block hooks", """
**What:** a forward hook on the patch embed and on each of the 12 transformer blocks reads that
module's CLS activations for the *same* images clean and degraded; CKA per module per severity
fills a block × severity matrix (the heatmap in the "Where it broke" section).

**Why per-block?** A global drift number tells you *that* the model moved; a per-block matrix
tells you *where*. Ours climbs from ~0.18 at the patch embed to ~0.81 in the last blocks — the
failure has an address.

**Guided-search honesty:** we looked at all 12 blocks and reported the largest, so there is a
selection effect — that is exactly why the unbiased-estimator recompute and the third-architecture
replication exist.
"""),
        ("Prove the mechanism with frequency surgery", """
**What:** a circular FFT mask splits each image in two: **low-pass** keeps only smooth structure
(large shapes, brightness — where luminance lives); **high-pass** keeps only edges and fine
texture. Same images, same severities, same probe — the only variable is which frequency band
survives.

**The prediction:** if the model runs on low-frequency luminance, low-pass should preserve
accuracy and high-pass should destroy it even when pixel statistics barely change.

**The result:** low-pass stays near clean (~0.92), high-pass collapses (~0.59) at every severity —
which is also why the same LoRA fix later transferred to blur and JPEG: those destroy the same
band.
"""),
        ("Try to make profiling cheaper — and accept the NO-GO", """
**What:** four cheap statistics — energy drop, mean shift, covariance shift, cosine drop — logged
next to CKA, then scored with **Spearman ρ**: a rank correlation in [-1, 1] that asks "when CKA
says corruption A is worse than B, does the cheap statistic agree?"

**The gate, committed to before looking:** ρ ≥ 0.80 on *every* corruption, strictly. Pass → we
could profile with pennies. Fail → we keep CKA and say so out loud.

**What happened:** amplitude statistics track JPEG (energy and structure move together there) but
fail on low-light (darkness shrinks amplitudes *without* ranking the structural late-block drift).
NO-GO on three architectures. The fallback is unapologetic: full CKA stays the profiler — and it
was never expensive: two forward passes, no labels, no backprop.
"""),
        ("Fix it with a 0.99% patch (LoRA + matched-noise grading)", """
**What:** keep the 22M-parameter backbone frozen and teach **24 tiny side-patches** to speak up
instead: LoRA adds a pair of small trainable matrices (rank 8, α16) next to every block's attention
projections — **221,184 trainable numbers = 0.99% of the model**. Analogy: don't repaint the
house; tape a thin film on the windows.

**The diet:** 5,000 training images, 70% darkened (severity 2–4, random per sample) + 30% clean,
plus flips, 10 epochs of AdamW — about 10 minutes on a free T4. The clean 30% is deliberate: it is
what stops the patch from trading away clean accuracy.

**Matched-noise grading:** darkness includes *random* noise, so if you corrupt the test set
separately for each model they sit different exams. The script of record pre-generates the
corrupted arrays **once** per severity (seed `1000 + severity`) and both models are graded on the
identical pixels — the before/after gap is real, not eval jitter.
"""),
        ("Stress the claim from every side", """
**What:** the headline claim survived five independent ways of trying to kill it —

1. **Seeds:** 3 arms × 3 seeds, because one lucky run proves nothing.
2. **Families:** blur, JPEG, contrast — each with its own ladder, its own profile, its own adapters.
3. **The skeptic's control:** retrain the classifier per severity to split feature loss from
   readout staleness.
4. **Real data:** ExDark's 7,363 real low-light photos, adapters transferred with zero retraining.
5. **A third architecture:** ViT-B/14 profiled at sublayer granularity (attention vs MLP).

**Why the trouble?** Each replication targets a *different* way the study could be wrong —
overfitting to one seed, one corruption, one model, one dataset, or one measurement instrument.
Better we break it here than a reviewer does.
"""),
    ]
    for i, (title, body) in enumerate(steps, 1):
        with st.expander(f"Step {i} — {title}"):
            st.markdown(body)

    st.caption("Exact parameters from docs/METHODS.md \u2014 verified against source. Every experiment in this "
               "presentation reads committed artifacts produced with these settings.")

    tab1, tab2, tab3, tab4 = st.tabs(["Backbone & data", "Corruptions", "Metrics & statistics", "The LoRA fix"])

    with tab1:
        backbone = pd.DataFrame([
            {"Item": "Model", "Value": "DINOv2 ViT-S/14 via torch.hub (dinov2_vits14)"},
            {"Item": "Parameters", "Value": "22,277,760 \u2014 frozen, except Phase-3 LoRA + head"},
            {"Item": "Input size", "Value": "224 \u00d7 224 (CIFAR-10 32\u00d732 upsampled), patch 14 \u2192 256 tokens + CLS"},
            {"Item": "Embedding used", "Value": "pooled CLS token, dim 384"},
            {"Item": "Normalization", "Value": "ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)"},
            {"Item": "Transform order", "Value": "ToTensor \u2192 Resize((224,224), antialias) \u2192 Normalize"},
            {"Item": "Mode", "Value": "model.eval() + torch.no_grad() throughout"},
        ])
        st.table(backbone)

        st.subheader("Data protocol")
        data = pd.DataFrame([
            {"Experiment": "Phase 1 (collapse)", "Source": "CIFAR-10 test", "n": 1000,
             "Sampling": "default_rng(42).choice(10000, 1000), no replacement"},
            {"Experiment": "Phase 2 (localize)", "Source": "CIFAR-10 test", "n": 500,
             "Sampling": "fresh rng(42) draw \u2014 independent, ~5% overlap by chance"},
            {"Experiment": "Phase 3 training pool", "Source": "CIFAR-10 train", "n": 5000,
             "Sampling": "rng(42) on the 50,000 train images (same RNG, consumed sequentially)"},
            {"Experiment": "Phase 3 test set", "Source": "CIFAR-10 test", "n": 1000,
             "Sampling": "identical indices to Phase 1 \u2192 baselines match exactly (0.9133)"},
        ])
        st.table(data)
        st.caption("All probe splits stratified 70/30 (random_state=42): original and LoRA probes receive the "
                   "identical test indices, so every before/after comparison is same-images. Phase 3 trains on the "
                   "train split \u2014 no test image is ever seen during adapter training.")

    with tab2:
        corr = pd.DataFrame([
            {"Severity": 0, "low_light brightness": 1.00, "low_light noise \u03c3": 0,
             "blur radius (px)": 0.0, "jpeg quality": "pristine", "contrast factor": 1.0},
            {"Severity": 1, "low_light brightness": 0.75, "low_light noise \u03c3": 2,
             "blur radius (px)": 0.5, "jpeg quality": "60", "contrast factor": 0.8},
            {"Severity": 2, "low_light brightness": 0.55, "low_light noise \u03c3": 4,
             "blur radius (px)": 1.0, "jpeg quality": "40", "contrast factor": 0.6},
            {"Severity": 3, "low_light brightness": 0.38, "low_light noise \u03c3": 6,
             "blur radius (px)": 2.0, "jpeg quality": "25", "contrast factor": 0.4},
            {"Severity": 4, "low_light brightness": 0.25, "low_light noise \u03c3": 9,
             "blur radius (px)": 3.5, "jpeg quality": "15", "contrast factor": 0.25},
            {"Severity": 5, "low_light brightness": 0.15, "low_light noise \u03c3": 13,
             "blur radius (px)": 5.5, "jpeg quality": "8", "contrast factor": 0.15},
        ])
        st.table(corr)
        st.caption("low_light: img * brightness + N(0, \u03c3) clipped to [0,255], noise i.i.d. per pixel/channel. "
                   "contrast: factor \u00d7 img + (1 \u2212 factor) \u00d7 scalar grayscale mean (documented deviation from the "
                   "Hendrycks & Dietterich per-channel formula). jpeg sev 0 returns the pristine image untouched.")

        freq = pd.DataFrame([
            {"Severity": 1, "low_pass cutoff (keep \u2264 r)": 0.90, "high_pass cutoff (keep > r)": 0.05},
            {"Severity": 2, "low_pass cutoff (keep \u2264 r)": 0.70, "high_pass cutoff (keep > r)": 0.10},
            {"Severity": 3, "low_pass cutoff (keep \u2264 r)": 0.50, "high_pass cutoff (keep > r)": 0.15},
            {"Severity": 4, "low_pass cutoff (keep \u2264 r)": 0.35, "high_pass cutoff (keep > r)": 0.20},
            {"Severity": 5, "low_pass cutoff (keep \u2264 r)": 0.20, "high_pass cutoff (keep > r)": 0.30},
        ])
        st.subheader("Frequency controls (circular FFT mask per channel)")
        st.table(freq)
        _img_grid([
            ("output/notebook1/severity_visual_check.png", "Severity ladder on real images (of record)"),
            ("output/notebook2/corruption_types.png", "The corruption zoo used across the study"),
        ])

    with tab3:
        st.subheader("Linear CKA (corrected formula)")
        st.code(
            "X = X - X.mean(0, keepdims=True)   # row-centered activations\n"
            "Y = Y - Y.mean(0, keepdims=True)\n"
            "hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2\n"
            "var1 = np.linalg.norm(X.T @ X, 'fro') ** 2\n"
            "var2 = np.linalg.norm(Y.T @ Y, 'fro') ** 2\n"
            "CKA  = hsic / (np.sqrt(var1 * var2) + 1e-8)",
            language="python",
        )
        _bullets([
            "Range [0, 1]; CKA(X, X) = 1. Applied per transformer block on CLS activations of the same images, clean vs degraded.",
            "**History:** the original implementation omitted the sqrt (squaring all values, sev-5 up to ~79.9). "
            "Ordering was preserved (squaring is monotonic), so the late-layer conclusion held; every number in this repo uses the corrected formula.",
            "**Unbiased estimator check:** recomputed layer \u00d7 severity matrix with the unbiased CKA (Kornblith et al. 2019, App. B). "
            "Allocations differ at 3 of 12 blocks by \u00b11 rank; blocks 9\u201311 hold max rank 8 under both estimators "
            "(block 10 sev-5 drop 0.815 \u2192 0.859) \u2014 the late-heavy structure is estimator-robust.",
            "**Cross-device certificate:** the Phase-2 matrix computed locally (CPU) and inside the Kaggle T4 kernel agrees to ~1e-6 \u2014 CPU and GPU artifacts are directly comparable.",
        ])

        st.subheader("Bootstrap confidence intervals")
        boot = pd.DataFrame([
            {"Parameter": "Resamples", "Value": "1,000 per severity"},
            {"Parameter": "Interval", "Value": "95th percentile (np.percentile at 2.5 / 97.5)"},
            {"Parameter": "Unit", "Value": "per-image correctness vector, resampled with replacement"},
            {"Parameter": "RNG", "Value": "np.random.default_rng(severity) \u2014 severity-indexed seed (bug fix; see Bugs)"},
        ])
        st.table(boot)

        st.subheader("Linear probe")
        probe = pd.DataFrame([
            {"Parameter": "Classifier", "Value": "sklearn LogisticRegression"},
            {"Parameter": "max_iter / C", "Value": "2,000 / 1.0 (inverse regularization)"},
            {"Parameter": "Train data", "Value": "clean severity-0 embeddings only"},
            {"Parameter": "Split", "Value": "70/30, stratified, seed 42 \u2014 identical test indices for both models"},
        ])
        st.table(probe)

    with tab4:
        place = pd.DataFrame([
            {"Item": "Targets", "Value": "attn.qkv + attn.proj in every block (24 modules)"},
            {"Item": "Rank / alpha", "Value": "r = 8, \u03b1 = 16 (scaling = \u03b1/r = 2.0)"},
            {"Item": "Trainable params", "Value": "221,184 / 22,277,760 = 0.99%"},
            {"Item": "Backbone", "Value": "frozen; classifier head (384 \u2192 10) also trains"},
        ])
        st.table(place)

        train = pd.DataFrame([
            {"Item": "Steps", "Value": "10 epochs \u00d7 780 steps = ~7,800 total"},
            {"Item": "Optimizer", "Value": "AdamW; adapters lr 5e-5, head lr 1e-3, weight_decay 0.01"},
            {"Item": "Batch / loss", "Value": "64, shuffled, num_workers 2; CrossEntropy"},
            {"Item": "Augmentation", "Value": "p=0.7 low_light sev ~ Uniform{2,3,4}; p=0.5 horizontal flip (train only)"},
            {"Item": "Runtime", "Value": "~10 min on a free-tier Colab/Kaggle T4"},
        ])
        st.table(train)

        st.subheader("Evaluation \u2014 matched-noise protocol")
        st.markdown("""
        - Corrupted test arrays precomputed **once** per severity (seeded `default_rng(1000 + severity)`) and fed
          **identically** to the original and adapted models \u2014 every before/after pair is image- and noise-matched.
        - Probe protocol identical to Phase 1 (clean embeddings, 70/30, seed 42).
        - Artifact of record: `colab_results/lora_run_fixed/` (post augmentation-fix, matched-noise).
        """)

        st.subheader("The 9 arms")
        arms = pd.DataFrame([
            {"Arm": "seed42/43/44_all", "Params": "221,184 (0.99%)", "Notes": "uniform rank-8, 3 seeds"},
            {"Arm": "late", "Params": "55,296 (0.25%)", "Notes": "rank-8 on blocks 9\u201311 only"},
            {"Arm": "drift \u00d7 3 seeds", "Params": "172,800 (0.78%)",
             "Notes": "per-block ranks \u221d CKA drop: {0:6, 1:6, 2:4, 3:5, 4:5, 5:5, 6:6, 7:7, 8:7, 9:8, 10:8, 11:8}"},
            {"Arm": "fullft \u00d7 3 seeds", "Params": "22,056,576 (100%)", "Notes": "backbone lr 1e-5, head lr 1e-3"},
        ])
        st.table(arms)
        st.caption("Cross-seed caveat: each run draws its own 1,000-image test set seeded by --seed, so rows with "
                   "different seeds differ in both init and test draw. Within any row, original vs adapted is image- and noise-matched.")


def _render_collapse():
    st.subheader(SECTION_TITLES["collapse"])
    st.markdown("""
    *How we did it.* We took 1,000 CIFAR-10 test images (a fixed, seeded draw), pushed them through
    the six-step "dimmer switch" (severity 0 = untouched → 5 = darkest), and asked the frozen model
    two questions at every level: **(1)** can a simple linear classifier still read the labels off
    its features? and **(2)** how far have those features themselves moved from their clean shape?
    Both questions are answered on the *same* images, so the two curves can be read against each
    other directly — accuracy loss versus representation drift, severity by severity.
    """)
    df = load_notebook1()

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(_plot_acc_drift(df, "How accuracy and embedding drift fall as darkness increases"))
    with c2:
        st.dataframe(df, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Clean (sev 0)", "0.913")
        col2.metric("Sev 5", "0.093", "chance = 0.10")

    _img("output/notebook1/dinov2_lowlight_results.png",
         "Figure of record: full accuracy + drift curve with bootstrapped 95% CIs (1,000 resamples per severity)")

    st.subheader("Findings")
    _bullets([
        "**H1 supported:** the drop is a **cliff**, not a slope \u2014 near-flat through sev 2, collapsing steeply 2\u21924.",
        "Accuracy loss tracks embedding drift **almost linearly** \u2014 the feature space moves; the model isn't 'confused'.",
        "At sev 5 the probe is barely above chance (chance = 0.10).",
        "Cosine drift tracks accuracy loss in every family tested later \u2014 this signature isn't low-light specific.",
    ])


def _render_localize():
    st.subheader(SECTION_TITLES["localize"])
    st.markdown("""
    *How we did it.* To find *where* the model breaks, we clipped a sensor (a forward hook) onto the
    patch embed and onto each of the 12 transformer blocks, then pushed the **same** 500 images
    through twice — once clean, once degraded — and compared each block's CLS activations with
    **linear CKA**, a score from 0 to 1 that answers "how much did the internal picture rearrange?"
    (1.0 = identical, 0 = completely reshuffled). No labels are involved: CKA only looks at the
    geometry of the activations. Doing that for every block at every severity gives the matrix
    below — and the pattern in it *is* the localization answer.
    """)
    cka = load_cka_matrix()

    c1, c2 = st.columns(2)
    with c1:
        _img("output/notebook2/layerwise_cka.png",
             "Notebook-2 figure of record: layer \u00d7 severity CKA matrix")
    with c2:
        sev = st.slider("Severity for the bar chart", min_value=1, max_value=5, value=5)
        st.pyplot(_plot_layerwise(cka, sev))
        d = cka[cka["severity"] == sev]
        top = d.loc[d["drop_from_clean"].idxmax()]
        st.caption(f"Max drift at severity {sev}: block {int(top['layer'])} (drop = {top['drop_from_clean']:.3f})")

    st.subheader("The full layer \u00d7 severity matrix")
    unbiased = st.checkbox("Use the unbiased CKA estimator (Kornblith App. B) instead of the biased one", value=False)
    src = load_cka_unbiased() if unbiased else cka
    piv = src.pivot_table(index="layer", columns="severity", values="drop_from_clean")
    piv.index = [f"b{int(i)}" for i in piv.index]
    which = "unbiased" if unbiased else "biased"
    st.pyplot(_plot_heatmap(piv, f"Drift by block \u00d7 severity \u2014 {which} CKA estimator"))
    st.caption("Late-heavy structure is estimator-robust: blocks 9\u201311 hold maximum drift under both estimators. "
               "The published drift-arm allocation changes by \u00b11 rank at 3 of 12 blocks between estimators (\u22121,536 params).")

    st.subheader("Findings")
    _bullets([
        "Late blocks (9\u201311) hold full drift; early blocks comparatively stable \u2014 the failure has an address.",
        "The localization finding is a property of frequency-destroying degradations, not of darkness specifically.",
        "Guided-bias caveat: the published allocation used the biased estimator; the unbiased recompute shifts 3 of 12 "
        "block ranks by one without changing the late-heavy structure (see the toggle above).",
        "Cross-device determinism: CPU- and T4-computed matrices agree to ~1e-6, certifying artifact comparability.",
    ])


def _render_mechanism():
    st.subheader(SECTION_TITLES["mechanism"])
    st.markdown("""
    *How we did it.* "Darkness breaks the model" is an observation; *why* it breaks needs surgery.
    So we split each corrupted image into two frequency halves with a circular FFT mask: a
    **low-pass** image that keeps only the smooth structure (large shapes, brightness — where
    luminance lives) and a **high-pass** image that keeps only edges and fine texture. Same images,
    same severities, same probe — the only thing that changed is which frequency band survives. If
    the model really runs on low-frequency luminance, the low-pass half should keep accuracy alive
    and the high-pass half should kill it even when nothing else changed.
    """)
    c1, c2 = st.columns(2)
    with c1:
        _img("output/notebook2/frequency_test.png",
             "Frequency control: low-pass preserves accuracy; high-pass destroys it")
    with c2:
        _bullets([
            "**Low-pass** images: near-clean accuracy (~0.92 at sev 1).",
            "**High-pass**: collapses (~0.59) at every severity.",
            "DINOv2 runs on low-frequency luminance; darkness removes exactly that.",
            "The localization finding is a property of frequency-destroying degradations, not of darkness specifically.",
        ])
        st.pyplot(_plot_acc_drift(load_notebook1(), "Same signature in the primary low-light data",
                                  y2="mean_cosine_sim_to_clean"))

    st.subheader("The same control on the other corruption families")
    tabs = st.tabs(["blur", "jpeg", "contrast"])
    fam_dirs = {"blur": "sessionE/nb2_blur", "jpeg": "sessionE/nb2_jpeg", "contrast": "kaggle_sessionF/nb2_contrast"}
    for tab, name in zip(tabs, fam_dirs):
        with tab:
            rel = Path("colab_results") / fam_dirs[name]
            _img_grid([
                (rel / "frequency_test.png", f"{name}: low-pass vs high-pass control"),
                (rel / "accuracy_curve.png", f"{name}: accuracy across severities (with 95% CIs)"),
            ])

    st.subheader("Findings")
    _bullets([
        "Low-frequency luminance is the load-bearing signal \u2014 keep it (low-pass) and accuracy survives; remove it "
        "(high-pass) and the model collapses even when pixel statistics stay put.",
        "This is the mechanism that ties the whole study together: every destructive corruption in the family set "
        "deletes or scrambles exactly this band, which is why cosine drift tracks accuracy loss everywhere.",
        "Because the mechanism is frequency, not 'darkness', the same LoRA fix generalizes to blur and JPEG (section 7).",
    ])


def _render_proxy():
    st.subheader(SECTION_TITLES["proxy"])
    st.markdown("""
    *How we did it (the honest negative-result track).* Full CKA profiling is already cheap — two
    forward passes per condition, no labels, no backprop — but could it be cheaper? We logged four
    statistics that need no CKA at all (**energy drop, mean shift, covariance shift, cosine drop**)
    and scored them against CKA with **Spearman ρ**: a rank correlation that asks, in plain words,
    "when CKA says corruption A is worse than B, does the cheap statistic agree?" The gate we
    committed to *before looking*: ρ ≥ 0.80 on **every** corruption, strictly. Passing would mean
    we could replace CKA everywhere; failing means we keep CKA and say so.
    """)
    agreement = load_drift_proxy_agreement()
    gate = agreement["gate"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Gate verdict", gate["verdict"].upper())
        st.metric("Best proxy", gate["best_proxy"])
        passing = gate["per_proxy"][gate["best_proxy"]]["corruptions_passing_rho"]
        st.metric("Corruptions passing \u03c1", ", ".join(passing) if passing else "none")
        st.metric("Strict pass", "Yes" if gate["per_proxy"][gate["best_proxy"]]["strict_pass"] else "No")
    with c2:
        txt = st.expander("Raw gate verdict (agreement_report.json)")
        txt.json(gate)

    st.subheader("Proxy \u03c1 across corruptions")
    proxy_rows = []
    for p, info in gate["per_proxy"].items():
        for corr, rho in info["rho"].items():
            proxy_rows.append({"proxy": p, "corruption": corr, "rho": rho})
    proxy_df = pd.DataFrame(proxy_rows)
    st.pyplot(_plot_proxy(proxy_df, "Spearman \u03c1 between each cheap proxy and CKA drift"))

    st.subheader("What the profiles look like (the artifact the gate reads)")
    _img_grid([
        ("output/drift_proxy/profile_heatmap.png", "Module \u00d7 condition drift heatmap (proxy harness of record)"),
        ("output/drift_proxy/profiles_low_light.png", "Per-module drift profile \u2014 low_light"),
        ("output/drift_proxy/profiles_blur.png", "Per-module drift profile \u2014 blur"),
        ("output/drift_proxy_dinov2_check/profile_heatmap.png", "DINOv2 re-check run (ViT-S re-confirmed)"),
    ])

    st.subheader("Findings")
    _bullets([
        "**Diagnosis:** CKA is scale-invariant; every proxy except cos_drop is scale-sensitive.",
        "Amplitude proxies (energy/mean/cov) rank jpeg drift fine (structural change correlates with energy amplitude).",
        "On low_light, **amplitude shrink \u2260 structural late-stage drift** \u2014 the drift CKA sees is geometric/structural, "
        "which amplitude statistics can't rank.",
        "`mean_shift` \u2248 0 at every 1\u00d71-conv downsample (Gaussian blur preserves DC) \u2014 hook sanity check passed.",
        "ViT-S re-check (n=300): NO-GO, energy \u03c1 \u2248 0.70. ViT-B (n=1000): best energy_drop \u03c1 = 0.876 jpeg / "
        "0.692 low_light; cos_drop 0.882 / 0.681 \u2014 **third-architecture NO-GO** (see section 10).",
        "**Fallback adopted:** full CKA stays the profiler. It's already forward-only (two passes, no labels, no "
        "backprop). The paper claim weakens to 'single cheap profiling pass' rather than 'cheap proxy'.",
    ])


def _render_fix():
    st.subheader(SECTION_TITLES["fix"])
    st.markdown("""
    *How we did it.* Rather than retraining 22 million weights — slow, risky, and likely to forget
    clean images — we left the backbone **frozen** and taught 24 tiny side-patches to speak up
    differently: LoRA (low-rank adaptation) adds a pair of small trainable matrices (rank 8, α16)
    next to every block's attention projections: **221,184 trainable numbers = 0.99% of the
    model**. Analogy: don't repaint the house — tape a thin film on the windows. The training diet
    deliberately mixes dark and clean (70% augmented images at severity 2–4, 30% clean, plus
    flips) so the patch learns darkness *without* trading away clean accuracy. And because darkness
    includes random noise, both the original and the patched model are graded on the **identical**
    pre-corrupted test arrays (seeded `1000 + severity`) — the before/after gap is real, not eval
    jitter.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean (sev 0\u20135)", "0.594 \u2192 0.818", "+22.4 pts")
    col2.metric("Worst case (sev 5)", "0.073 \u2192 0.377", "5.1\u00d7")
    col3.metric("Trainable params", "0.99%", "221,184 of 22.3M")
    col4.metric("Training cost", "~10 min", "free-tier T4, 10 epochs")

    st.subheader("Run of record (post augmentation-fix, matched-noise eval)")
    _img_grid([
        ("colab_results/lora_run_fixed/lora_training_curves.png",
         "Adapter training curves \u2014 train/val loss and accuracy, 10 epochs"),
        ("colab_results/lora_run_fixed/lora_vs_orig_accuracy.png",
         "Adapted vs original model at every severity (the headline figure)"),
    ])

    st.subheader("What was trained")
    _bullets([
        "**Single-run headline (run of record):** LoRA rank 8, \u03b1 16, on `attn.qkv` + `attn.proj` in all 12 blocks "
        "(24 modules); 221,184 / 22,277,760 = 0.99%.",
        "Backbone frozen; classifier head (384 \u2192 10) also trains.",
        "5,000 CIFAR-10 train images, 70% low-light-augmented (sev 2\u20134 random, horizontal flips, per-sample "
        "`default_rng(seed=idx)`).",
        "10 epochs, AdamW, batch 64, adapter lr 5e-5 / head lr 1e-3, weight_decay 0.01.",
        "Matched-noise eval: corrupted test arrays precomputed **once** per severity (seeded `default_rng(1000 + "
        "severity)`) and fed identically to both models \u2014 image- and noise-matched comparison.",
        "**Mean** 0.594 \u2192 0.818 (+0.224); **worst-case 5.1\u00d7** (0.073 \u2192 0.377); clean accuracy **even rises**.",
    ])
    st.caption("Older pre-fix artifacts are retained in colab_results/lora_run/ for comparison only \u2014 everything "
               "quoted in this presentation comes from lora_run_fixed/ (see methods, tab 'The LoRA fix').")

    st.subheader("Findings")
    _bullets([
        "A 0.99%-parameter adapter recovers most of the collapse without touching the frozen backbone.",
        "Clean accuracy doesn't trade off \u2014 it rises slightly (the clean 30% of the mix acts as a stabilizer).",
        "The single run is the headline; section 6 is where the statistically defensible claim lives (3 seeds \u00d7 3 arms).",
    ])


def _render_grid():
    st.subheader(SECTION_TITLES["grid"])
    st.markdown("""
    *How we did it.* One good run can be luck — a friendly seed, a lucky shuffle. So we ran the
    three serious candidates — uniform rank-8 LoRA, drift-weighted LoRA (ranks scaled to each
    block's measured drift), and full fine-tuning — with **three seeds each** (42/43/44), plus a
    late-blocks-only arm as a control for "just patch where the drift is". Every arm eats the same
    5,000-image diet and is graded with the same matched-noise protocol; the error bars below are
    the spread across seeds. Each seed also draws its own test set, so rows differ in init *and*
    data draw — but within any row, original-vs-adapted is still the same images.
    """)

    df = load_seed_analysis()
    grid = df[df["status"] == "ok"].copy()
    grid["label"] = grid["arm"].map({"uniform": "Uniform LoRA r8",
                                     "drift": "Drift-weighted ranks",
                                     "fullft": "Full fine-tuning"})
    grid = grid.sort_values("group")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(grid[["label", "seed", "mean_acc", "sev5_acc", "clean_acc"]],
                     use_container_width=True, hide_index=True)
    with c2:
        st.pyplot(_plot_seed_grid(grid, "mean_acc"))

    st.caption("Parity: drift 0.825 \u00b1 0.016, uniform 0.827 \u00b1 0.012, full FT 0.831 \u00b1 0.019 \u2014 ranges fully "
               "overlap. Drift trains 0.78% of params; full FT buys nothing on mean and costs clean accuracy.")

    st.subheader("Per-arm training curves and before/after accuracy (seed of record per tab)")
    arms_tabs = st.tabs(["Uniform LoRA (3 seeds)", "Drift-weighted (3 seeds)",
                         "Full fine-tuning (3 seeds)", "Late-only (negative result)"])
    arm_dirs = {
        "Uniform LoRA (3 seeds)": ["seed42_all", "seed43_all", "seed44_all"],
        "Drift-weighted (3 seeds)": ["drift", "drift_seed43", "drift_seed44"],
        "Full fine-tuning (3 seeds)": ["fullft", "fullft_seed43", "fullft_seed44"],
        "Late-only (negative result)": ["late"],
    }
    for tab, name in zip(arms_tabs, arm_dirs):
        with tab:
            pairs = []
            for d in arm_dirs[name]:
                rel = Path("colab_results") / "sessionD" / d
                pairs.append((rel / "lora_training_curves.png", f"{d}: training curves"))
                pairs.append((rel / "lora_vs_orig_accuracy.png", f"{d}: adapted vs original"))
            _img_grid(pairs)

    st.subheader("Findings")
    _bullets([
        "**Parity at a fraction of the cost** \u2014 drift 0.825 \u00b1 0.016, uniform 0.827 \u00b1 0.012, full FT 0.831 \u00b1 0.019; "
        "all ranges fully overlap.",
        "**Full FT buys nothing on mean and costs clean accuracy** \u2014 its mean is statistically indistinguishable "
        "from both LoRA arms; clean accuracy worst of all adapted arms (seed mean 0.960 vs 0.972\u20130.973).",
        "**Late-only is a useful negative result** \u2014 drift-concentrated blocks only, 0.25% of params, reaches just 0.686.",
        "**Seed spread real and quantified** \u2014 SDs \u00b11.2 / \u00b11.6 / \u00b11.9 points "
        "(ranges 0.815\u20130.839 / 0.811\u20130.842 / 0.810\u20130.846).",
    ])


def _render_families():
    st.subheader(SECTION_TITLES["families"])
    st.markdown("""
    *How we did it.* If the story is "fix the localized drift", it should work for **any**
    corruption that breaks the model the same way — not just darkness. So we repeated the whole
    recipe for three more families: build each family's own severity ladder (Gaussian blur radii,
    JPEG quality steps, contrast factors), measure its accuracy curve and CKA drift profile, then
    train family-specific LoRA patches — both drift-weighted and uniform — through the exact same
    diet and matched-noise grading. The blur arms were salvaged: their Colab eval logs truncated,
    so we re-graded the saved checkpoints locally — and the re-graded original column reproduces
    Phase 1's curve *exactly* (0.913 / 0.873 / 0.620 / 0.387 / 0.220 / 0.143), which is our
    comparability certificate.
    """)

    fragility = pd.DataFrame([
        {"Corruption": "low_light (ref)", "Sev-5 accuracy": 0.087, "Cos-sim to clean": 0.161, "Profile": "cliff between 2\u20134"},
        {"Corruption": "blur", "Sev-5 accuracy": 0.143, "Cos-sim to clean": 0.201, "Profile": "cliff between 1\u20133 (earliest)"},
        {"Corruption": "JPEG", "Sev-5 accuracy": 0.273, "Cos-sim to clean": 0.321, "Profile": "steep+early (0.73 already at sev 1)"},
        {"Corruption": "contrast", "Sev-5 accuracy": 0.840, "Cos-sim to clean": 0.840, "Profile": "mild, near-linear decline"},
    ])
    st.subheader("How fragile is each family (original model)?")
    st.dataframe(fragility, use_container_width=True, hide_index=True)

    fam = pd.DataFrame([
        {"Arm": "JPEG, drift-weighted", "Before \u2192 After": "0.586 \u2192 0.779", "\u0394": "+19.3", "Notes": "sev-5 +25.7; kaggle_sessionF/jpeg_drift"},
        {"Arm": "JPEG, uniform", "Before \u2192 After": "trained, eval complete", "\u0394": "\u2014", "Notes": "sessionE/jpeg_uniform; folded into analysis"},
        {"Arm": "blur, drift-weighted", "Before \u2192 After": "0.526 \u2192 0.703", "\u0394": "+17.7", "Notes": "sev-5 0.143\u21920.303; re-eval from retained adapters"},
        {"Arm": "blur, uniform r8", "Before \u2192 After": "0.526 \u2192 0.698", "\u0394": "+17.2", "Notes": "sev-5 0.143\u21920.270; drift edges worst-case"},
        {"Arm": "contrast, uniform r8", "Before \u2192 After": "0.885 \u2192 0.960", "\u0394": "+7.5", "Notes": "mild corruption, mild recovery"},
        {"Arm": "contrast, drift-weighted", "Before \u2192 After": "0.885 \u2192 0.954", "\u0394": "+6.9", "Notes": "graceful where drift-weight has no signal"},
    ])
    st.subheader("Does the fix generalize?")
    st.dataframe(fam, use_container_width=True, hide_index=True)

    tabs = st.tabs(["blur", "jpeg", "contrast"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(_plot_two_arm(load_bridge_drift(), "original", "adapted",
                                    "Blur \u2014 drift-weighted arm, re-evaluated from retained adapters",
                                    label_a="Original model", label_b="Drift-weighted LoRA"))
        with c2:
            st.pyplot(_plot_two_arm(load_bridge_uniform(), "original", "adapted",
                                    "Blur \u2014 uniform arm, re-evaluated from retained adapters",
                                    label_a="Original model", label_b="Uniform LoRA"))
        st.caption("Comparability certificate: the re-evaluated original column reproduces Phase 1 exactly "
                   "(0.913 / 0.873 / 0.620 / 0.387 / 0.220 / 0.143).")
        _img_grid([
            ("colab_results/sessionE/blur_drift/lora_training_curves.png", "blur drift: training curves (Colab T4)"),
            ("colab_results/sessionE/blur_drift/lora_vs_orig_accuracy.png", "blur drift: adapted vs original"),
            ("colab_results/sessionE/blur_uniform/lora_training_curves.png", "blur uniform: training curves"),
            ("colab_results/sessionE/blur_uniform/lora_vs_orig_accuracy.png", "blur uniform: adapted vs original"),
            ("colab_results/sessionE/nb2_blur/accuracy_curve.png", "blur: Phase-2 accuracy curve with 95% CIs"),
            ("colab_results/sessionE/nb2_blur/frequency_test.png", "blur: frequency control"),
        ])

    with tabs[1]:
        _img_grid([
            ("colab_results/kaggle_sessionF/jpeg_drift/lora_training_curves.png", "jpeg drift: training curves (Kaggle T4)"),
            ("colab_results/kaggle_sessionF/jpeg_drift/lora_vs_orig_accuracy.png", "jpeg drift: adapted vs original (+19.3 mean)"),
            ("colab_results/sessionE/jpeg_uniform/lora_training_curves.png", "jpeg uniform: training curves"),
            ("colab_results/sessionE/jpeg_uniform/lora_vs_orig_accuracy.png", "jpeg uniform: adapted vs original"),
            ("colab_results/sessionE/nb2_jpeg/accuracy_curve.png", "jpeg: Phase-2 accuracy curve with 95% CIs"),
            ("colab_results/sessionE/nb2_jpeg/frequency_test.png", "jpeg: frequency control"),
        ])

    with tabs[2]:
        _img_grid([
            ("colab_results/kaggle_sessionF/contrast_drift/lora_training_curves.png", "contrast drift: training curves"),
            ("colab_results/kaggle_sessionF/contrast_drift/lora_vs_orig_accuracy.png", "contrast drift: adapted vs original (+6.9)"),
            ("colab_results/kaggle_sessionF/contrast_uniform/lora_training_curves.png", "contrast uniform: training curves"),
            ("colab_results/kaggle_sessionF/contrast_uniform/lora_vs_orig_accuracy.png", "contrast uniform: adapted vs original (+7.5)"),
            ("colab_results/kaggle_sessionF/nb2_contrast/accuracy_curve.png", "contrast: Phase-2 accuracy curve with 95% CIs"),
            ("colab_results/kaggle_sessionF/nb2_contrast/frequency_test.png", "contrast: frequency control"),
        ])

    st.subheader("Findings")
    _bullets([
        "**Remediation generalizes** \u2014 drift-weighted LoRA delivers large gains on corruptions never designed for "
        "(JPEG +19.3, blur +17.7) using each family's own CKA profile.",
        "**Allocation rule honest and repeats the low-light signature** \u2014 contrast: profile flat, drift-weighting "
        "matches uniform (+6.9 vs +7.5).",
        "**Gain size tracks fragility** \u2014 low_light +22.4 > JPEG +19.3 \u2248 blur +17.7 > contrast +7.",
        "Caveats: family arms are single-seed (seed 42); deltas exceed seed noise comfortably, but within-family "
        "parity claims are single-seed.",
        "Blur arms trained on Colab T4 with truncated eval logs were salvaged by re-evaluating the retained "
        "`lora_adapters.pt` checkpoints locally \u2014 comparability certified against the Phase-1 curve above.",
    ])


def _render_readout():
    st.subheader(SECTION_TITLES["readout"])
    st.markdown("""
    *How we did it (the "is it just the classifier?" control).* A skeptic could argue: the features
    moved, fine — but maybe a *fresh* classifier would cope, and the real bug is our stale probe.
    So we split the question in two arms on the same folds and the same 1,000 images: **Arm A**
    keeps the severity-0 probe frozen across all severities (measures how stale the readout gets),
    **Arm B** retrains a probe per severity on degraded versions of the *same training fold*
    (measures how much label information survived in the features themselves). The two arms are
    compared per image with a paired permutation test — 5,000 shuffles; in plain words: swap the
    two arms' grades 5,000 times and see how often luck alone produces a gap this big.
    """)
    df = load_readout()
    summary = load_readout_summary()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(_plot_readout(df))
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Sev-4 recovery", "+25.3 pp", "p = 0.0002")
        m2.metric("Sev-5 ceiling", "0.377", "vs 0.913 clean")
        st.markdown("""
        - **Arm A** = probe fixed across severities; **Arm B** = probe retrained per severity on the same train
          fold (strict same-fold control).
        - Paired permutation test (5,000 permutations, seed 42), Wilson 95% CIs, readout-collapse metrics.
        """)

    st.subheader("Per-severity decomposition (from summary.json)")
    rows = []
    for s in summary["severities"]:
        rows.append({
            "Severity": s["severity"],
            "Arm A fixed": round(s["fixed_acc"], 3),
            "Arm B adapted": round(s["adapted_acc"], 3),
            "\u0394 recovery (pp)": round(s["recovery_pp"], 1),
            "p (paired perm.)": round(s["p_paired"], 4),
            "top1_share (A)": round(s["fixed_top1_share"], 2),
            "entropy_ratio (A)": round(s["fixed_entropy_ratio"], 2),
            "dominant class (A)": s["fixed_dominant"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Findings")
    _bullets([
        "**Readout collapse signature (Arm A):** top1_share 0.14 (sev 2, dominant *deer*) \u2192 0.73 (sev 5, "
        "dominant *frog*); entropy_ratio 1.00 \u2192 0.26 \u2014 predictions concentrate on a single wrong class.",
        "Confirms upstream's ViT-B pilot (+22.5 pp at sev-4, n=120) at proper power on ViT-S (n=1000).",
        "**Readout staleness real and large in the mid-to-deep regime** \u2014 but the majority of loss stays in the "
        "features: even a perfect re-trained readout reaches only 0.377 at sev 5 vs 0.913 clean.",
        "Adapted probe worse at sev 1\u20132 (\u22121 to \u22122 pts, n.s.) \u2014 retraining the readout on mildly degraded data "
        "costs clean-fold generalization before drift is real.",
        "Interpretation: this is *why* the LoRA fix targets features \u2014 the head is at most a third of the story.",
    ])


def _render_exdark():
    st.subheader(SECTION_TITLES["exdark"])
    st.markdown("""
    *How we did it.* Synthetic extreme darkening is a deliberately harsh stress test — a fair
    reviewer asks: "but do **real** dark photos behave like that?" So we ran the identical
    frozen-model + linear-probe pipeline on **ExDark**: 7,363 real low-light photographs across 12
    object classes, binned by their actual measured luminance instead of by our synthetic knob.
    Then two zero-retraining transfers: **CLAHE** (pure brightening — tests whether pixel
    statistics alone help) and our **synthetic-dark LoRA adapters** (trained only on synthetic dark
    CIFAR, applied as-is). Whatever the adapters gain on real images is genuine transfer; the
    distance between the synthetic +21-point gain and the real gain is the sim-to-real gap —
    measured, not hand-waved.
    """)
    base = load_exdark_summary("exdark_baseline")
    drift = load_exdark_summary("exdark_transfer_drift")
    uniform = load_exdark_summary("exdark_transfer_uniform")

    base_raw = float(base.loc[base["pass"] == "raw", "accuracy"].iloc[0])
    clahe = float(base.loc[base["pass"] == "clahe", "accuracy"].iloc[0])
    drift_tr = float(drift.loc[drift["pass"] == "lora_transfer", "accuracy"].iloc[0])
    unif_tr = float(uniform.loc[uniform["pass"] == "lora_transfer", "accuracy"].iloc[0])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "7,363 images", "12 classes")
    col2.metric("Frozen accuracy", f"{base_raw:.3f}", "flat across luminance")
    col3.metric("CLAHE", f"{clahe - base_raw:+.3f}", "brightening alone: ~nothing")
    col4.metric("Synthetic-dark LoRA transfer", f"{drift_tr - base_raw:+.3f}",
                f"uniform {unif_tr - base_raw:+.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(_plot_darkness_curve(load_exdark_darkness("exdark_baseline")))
    with c2:
        st.pyplot(_plot_class_acc(load_exdark_class("exdark_baseline")))

    tab1, tab2, tab3 = st.tabs(["Figures of record", "Result tables", "Transfer experiments"])
    with tab1:
        _img_grid([
            ("output/exdark_baseline/exdark_summary.png", "ExDark summary figure (of record)"),
            ("output/exdark_baseline/darkness_curve.png", "Darkness-axis curve (of record)"),
        ])
    with tab2:
        st.dataframe(base, use_container_width=True, hide_index=True)
        st.dataframe(load_exdark_class("exdark_baseline"), use_container_width=True, hide_index=True)
    with tab3:
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Drift-weighted transfer**")
            st.dataframe(drift, use_container_width=True, hide_index=True)
            _img("output/exdark_transfer_drift/exdark_summary.png", "Transfer summary \u2014 drift arm")
        with tc2:
            st.markdown("**Uniform transfer**")
            st.dataframe(uniform, use_container_width=True, hide_index=True)
            _img("output/exdark_transfer_uniform/exdark_summary.png", "Transfer summary \u2014 uniform arm")

    st.subheader("Findings")
    _bullets([
        "**ExDark:** 7,363 images, 12 classes; frozen accuracy 0.725 **flat across the luminance axis** \u2014 real "
        "low light is far gentler than our synthetic severity 5.",
        "CLAHE (+0.001) does nothing on its own \u2014 pixel-statistics brightening isn't the fix; representation is.",
        "Synthetic-dark drift-LoRA transfer +1.9 pts (uniform +1.6) \u2014 only ~9% of the +21-point synthetic-axis gain.",
        "**Sim-to-real gap quantified, not assumed** \u2014 the synthetic axis is a worst-case stress test, and the "
        "adapters do transfer, just modestly.",
    ])


def _render_vitb():
    st.subheader(SECTION_TITLES["vitb"])
    st.markdown("""
    *How we did it.* "One model, one seed" is a fragile claim — so we moved to a third, larger
    architecture, **DINOv2 ViT-B/14 (86M params)**, and profiled it at *finer grain than before*:
    not just each whole block, but every block's **attention** and **MLP** sublayers individually —
    37 modules per condition — under two corruptions × five severities. Everything ran on a free
    Kaggle T4 in 259.5 seconds using the same label-free protocol: two forward passes per
    condition, hooks on each module, CLS-token pooling, matched-noise corruption (`1000 +
    severity`). The finer grain matters: the attention-vs-MLP asymmetry is invisible at
    whole-block resolution.
    """)
    profiles = load_vitb_profiles()
    agreement = load_vitb_agreement()
    gate = agreement["gate"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "DINOv2 ViT-B/14", "86M params, frozen")
    col2.metric("Modules probed", "37", "patch_embed + 12\u00d7(block, attn, mlp)")
    col3.metric("Gate verdict", gate["verdict"].upper())
    col4.metric("Compute", "259.5 s", "Kaggle T4, 2 fwd passes/condition")

    corr = st.radio("Corruption", ["low_light", "jpeg"], horizontal=True,
                    key="vitb_corr")

    st.subheader("Block-level drift heatmap (unbiased CKA drop, all severities)")
    blk = profiles[profiles["module"].str.fullmatch(r"blocks\.\d+")].copy()
    blk["block"] = blk["module"].str.extract(r"blocks\.(\d+)").astype(int)
    hc1, hc2 = st.columns(2)
    with hc1:
        piv = blk[blk["corruption"] == "low_light"].pivot_table(
            index="block", columns="severity", values="cka_unbiased_drop")
        piv.index = [f"b{int(i)}" for i in piv.index]
        st.pyplot(_plot_heatmap(piv, "ViT-B/14 \u2014 low_light: block \u00d7 severity", figsize=(6.4, 4.6)))
    with hc2:
        pivj = blk[blk["corruption"] == "jpeg"].pivot_table(
            index="block", columns="severity", values="cka_unbiased_drop")
        pivj.index = [f"b{int(i)}" for i in pivj.index]
        st.pyplot(_plot_heatmap(pivj, "ViT-B/14 \u2014 jpeg: block \u00d7 severity", figsize=(6.4, 4.6)))

    b5 = blk[blk["severity"] == 5].pivot_table(index="block", columns="corruption", values="cka_unbiased_drop")
    st.caption(
        f"Sev-5 headline (unbiased drop): low_light b0 {b5.loc[0, 'low_light']:.3f} \u2192 "
        f"b9/b10/b11 {b5.loc[9, 'low_light']:.3f}/{b5.loc[10, 'low_light']:.3f}/{b5.loc[11, 'low_light']:.3f}; "
        f"jpeg b0 {b5.loc[0, 'jpeg']:.3f} \u2192 {b5.loc[9, 'jpeg']:.3f}/{b5.loc[10, 'jpeg']:.3f}/{b5.loc[11, 'jpeg']:.3f}. "
        "Late-heavy on the third architecture \u2014 replication holds.")

    st.subheader("Sublayer finding: MLP drifts more than attention")
    st.pyplot(_plot_vitb_sublayers(profiles, sev=5, corr=corr))

    sub = profiles[(profiles["severity"] == 5) & (profiles["corruption"] == corr) &
                   profiles["module"].str.fullmatch(r"blocks\.\d+\.(attn|mlp)")].copy()
    sub["block"] = sub["module"].str.extract(r"blocks\.(\d+)").astype(int)
    sub["kind"] = sub["module"].str.extract(r"\.(attn|mlp)")
    piv = sub.pivot_table(index="block", columns="kind", values="cka_unbiased_drop").sort_index()
    tbl = pd.DataFrame({"block": piv.index,
                        "attn": piv["attn"].round(3),
                        "mlp": piv["mlp"].round(3)})
    tbl["mlp \u2212 attn"] = (tbl["mlp"] - tbl["attn"]).round(3)
    tbl["mlp > attn"] = tbl["mlp \u2212 attn"] > 0
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    n_wins = int(tbl["mlp > attn"].sum())
    st.caption(f"{corr}, severity 5, unbiased CKA drop: MLP > attention in {n_wins}/12 blocks "
               f"(max gap {tbl['mlp \u2212 attn'].max():.3f}). The claim is computed live from the committed CSV.")

    st.subheader("Per-severity profile figures (of record)")
    sev = st.slider("Severity", min_value=1, max_value=5, value=5, key="vitb_sev")
    _img(f"output/vitb_profile/profiles_{corr}@sev{sev}.png",
         f"Per-module drift profiles \u2014 {corr}, severity {sev}")

    with st.expander("Raw gate verdict (agreement_report.json)"):
        st.json(gate)

    st.subheader("Findings")
    _bullets([
        "**Late-heavy block profile replicates** on an 86M-parameter backbone: drift climbs from the patch embed "
        "to the final blocks under both corruptions.",
        "**Sublayer finding invisible to ViT-S whole-block CKA:** MLP sublayers drift more than attention sublayers "
        "in all 12 blocks under both corruptions (low_light gaps 0.015\u20130.159, max at b2; jpeg gaps 0.017\u20130.108, max at b1).",
        "**Gate NO-GO with the scale-invariance signature replicated:** best \u03c1 jpeg 0.876 (energy_drop) / 0.882 "
        "(cos_drop) vs low_light 0.692 / 0.681 \u2014 third architecture, same failure mode.",
        "**Protocol:** 1,000 CIFAR-10 test images, seed 42, matched-noise rng 1000+sev; low_light + jpeg, sev 1\u20135; "
        "two forward passes per condition, no labels, no backprop.",
        "**Caveat for consumers:** the drift CSVs key `layer` by profiled-module row (0\u201336). The v9 LoRA consumer "
        "expects dense 0..11 block indices \u2014 select the 12 `blocks.k` rows and re-index before `--drift-csv` use.",
    ])


def _render_bugs():
    st.subheader(SECTION_TITLES["bugs"])
    st.markdown("""
    *Why we show the bugs.* Four bugs slipped into the work at different points — and every one was
    caught by a **sanity check** (metric bounds, unexpected stability, static validation,
    eval-matching) *before* any conclusion trusted the numbers. Publishing them is the point: a
    result is only as good as the checks that could have killed it, and each fix left the pipeline
    stronger — correct formula, actually-independent resamples, ordered definitions, isolated
    worker RNGs.
    """)
    bugs = pd.DataFrame([
        {"#": "1", "Bug": "CKA sqrt bug",
         "What": "Invalid metric (normalized by var1*var2 instead of sqrt(var1*var2)) \u2014 all values inflated (sev-5 up to 79.9).",
         "Fix": "Correct formula: CKA = HSIC / sqrt(HSIC_XX * HSIC_YY) + 1e-8. Ordering preserved (squaring monotonic), so late-layer conclusion held; absolute values meaningless. All reported numbers now corrected."},
        {"#": "2", "Bug": "Bootstrap same-seed",
         "What": "Re-seeded default_rng(0) inside the severity loop \u2192 every severity resampled identically \u2192 misleadingly-similar CI bands.",
         "Fix": "Fixed: per-severity seeds (np.random.default_rng(severity)). Lesson: randomness must be actually independent across conditions compared."},
        {"#": "3", "Bug": "Port-time regression",
         "What": "head_ids referenced before head_params defined; caught by static validation pre-commit.",
         "Fix": "Lesson: script-vs-notebook drift is real; order-of-definition bugs hide easily in notebook cells."},
        {"#": "4", "Bug": "Worker-correlated augmentation + unmatched eval noise",
         "What": "NumPy global RNG inside __getitem__ with num_workers=2 (workers fork without re-seeding); eval noise drawn independently for LoRA vs original pass \u2192 jitter.",
         "Fix": "Fixes: per-sample default_rng(seed=idx) for training, precomputed seeded corrupted arrays for eval."},
    ])
    st.dataframe(bugs, use_container_width=True, hide_index=True)
    _bullets([
        "Each bug caught and fixed **before results were trusted** \u2014 no number in this presentation rests on a known-broken artifact.",
        "Each teaches a reproducibility lesson: metric sanity checks, independent resampling, script-vs-notebook "
        "ordering, worker RNG isolation.",
        "The re-checks are visible in the artifacts: corrected CKA matrices, severity-indexed bootstrap bands, "
        "`lora_run_fixed/` (matched-noise run of record), per-sample seeded augmentation.",
    ])


def _gallery_bucket(rel: Path) -> str:
    s = str(rel)
    if "notebook1" in s:
        return "Phase 1 \u00b7 the collapse"
    if "nb2_" in s or "notebook2" in s:
        return "Phase 2 \u00b7 localization & mechanism"
    if "lora_run" in s:
        return "Phase 3 \u00b7 LoRA run of record"
    if "sessionD" in s:
        return "Phase 3 \u00b7 the 9-arm grid"
    if "sessionE" in s:
        return "Session E \u00b7 blur & jpeg"
    if "sessionF" in s:
        return "Session F \u00b7 jpeg & contrast"
    if "drift_proxy" in s:
        return "Track 1 \u00b7 proxy gate"
    if "exdark" in s:
        return "ExDark \u00b7 real darkness"
    if "vitb" in s:
        return "ViT-B/14 \u00b7 cross-architecture"
    return "Other"


GALLERY_BUCKETS = [
    "Phase 1 \u00b7 the collapse",
    "Phase 2 \u00b7 localization & mechanism",
    "Phase 3 \u00b7 LoRA run of record",
    "Phase 3 \u00b7 the 9-arm grid",
    "Session E \u00b7 blur & jpeg",
    "Session F \u00b7 jpeg & contrast",
    "Track 1 \u00b7 proxy gate",
    "ExDark \u00b7 real darkness",
    "ViT-B/14 \u00b7 cross-architecture",
    "Other",
]


def _render_gallery():
    st.subheader(SECTION_TITLES["gallery"])
    st.caption("Every committed PNG in output/ and colab_results/, grouped by phase \u2014 84 figures. "
               "Use Ctrl+F in the browser to search captions.")

    rels = []
    for base in (OUTPUT, COLAB):
        for p in sorted(base.rglob("*.png")):
            rel = p.relative_to(PROJECT)
            if rel.name.startswith("__static"):
                continue
            rels.append(rel)

    groups: dict = {}
    for r in rels:
        groups.setdefault(_gallery_bucket(r), []).append(r)
    names = [b for b in GALLERY_BUCKETS if b in groups]
    names += sorted(set(groups) - set(GALLERY_BUCKETS))

    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            st.caption(f"{len(groups[name])} figures")
            _img_grid([(r, str(r)) for r in groups[name]])


# ---------------------------------------------------------------------------
# Dispatch, static render mode, navigation
# ---------------------------------------------------------------------------

RENDERERS = {
    "home": _render_home,
    "methods": _render_methods,
    "collapse": _render_collapse,
    "localize": _render_localize,
    "mechanism": _render_mechanism,
    "proxy": _render_proxy,
    "fix": _render_fix,
    "grid": _render_grid,
    "families": _render_families,
    "readout": _render_readout,
    "exdark": _render_exdark,
    "vitb": _render_vitb,
    "bugs": _render_bugs,
    "gallery": _render_gallery,
}


def render_static_section(section: str = "home") -> int:
    """Render one section to a PNG using matplotlib directly (no Streamlit server).

    Sets PRESENTATION_STATIC=1 PRESENTATION_SECTION=<section> to use.
    Output: __static_<section>.png next to this script (CWD-independent).
    """
    plt.ioff()
    fn = RENDERERS.get(section)
    if fn is None:
        raise ValueError(f"unknown static section: {section!r}")

    orig_pyplot = st.pyplot

    def _capture(fig=None, **kw):
        f = fig if fig is not None else plt.gcf()
        out = PROJECT / f"__static_{section}.png"
        f.savefig(out, dpi=110)
        plt.close(f)
        return f

    st.pyplot = _capture
    try:
        fn()
    finally:
        st.pyplot = orig_pyplot
    return 0


def main():
    if os.environ.get("PRESENTATION_STATIC") == "1":
        raise SystemExit(render_static_section(os.environ.get("PRESENTATION_SECTION", "home")))

    if os.environ.get("PRESENTATION_SMOKE") == "1":
        """Render every section headlessly; exit 0 only if all render clean."""
        failed = []
        for name in SECTION_ORDER:
            try:
                render_static_section(name)
                print(f"  {name} OK")
            except Exception as exc:  # noqa: BLE001 - smoke must keep sweeping
                failed.append(name)
                print(f"  {name} FAIL: {exc}", file=sys.stderr)
        if failed:
            print(f"SMOKE FAIL: {len(failed)}/{len(SECTION_ORDER)} sections: {failed}", file=sys.stderr)
            raise SystemExit(1)
        print(f"SMOKE OK: all {len(SECTION_ORDER)} sections rendered")
        raise SystemExit(0)

    env_sec = os.environ.get("PRESENTATION_SECTION")
    default = env_sec if env_sec in SECTION_ORDER else "home"
    if "nav" not in st.session_state:
        st.session_state["nav"] = default

    st.sidebar.title("DINOv2 low-light study")
    st.sidebar.caption("11 phases + methods, bugs, gallery \u2014 reads committed artifacts only.")
    section = st.sidebar.radio("Sections", SECTION_ORDER, format_func=lambda k: SECTION_LABELS[k], key="nav")
    if env_sec in SECTION_ORDER:
        section = env_sec
    st.session_state["section"] = section

    RENDERERS[section]()

    st.divider()
    st.caption("Run: `streamlit run apps/presentation.py` \u2014 reads committed artifacts directly; "
               "no new experiments required. Values match the CSV/PNG artifacts of record.")


if __name__ == "__main__":
    main()

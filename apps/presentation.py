# ===========================================================================
def _render_vitb():
    st.subheader(SECTION_TITLES["vitb"])
    df = load_vitb_drift5()
    agreement = load_vitb_agreement()
    gate = agreement["gate"]

    # Reproduce the sublayer-drift table (mlp > attn in all 12 blocks, sev 5)
    sev5 = df[df["severity"] == 5]
    sub = sev5[sev5["module"].str.contains(r"\.attn|\.mlp", regex=True)]
    agg = (sub.groupby(["layer", "module"])["drop_from_clean"]
             .mean().unstack().reset_index())
    agg.columns = ["block", "attn", "mlp"]
    agg["diff (mlp - attn)"] = (agg["mlp"] - agg["attn"]).round(3)
    agg["mlp > attn"] = agg["diff (mlp - attn)"] > 0
    agg = agg.sort_values("block").reset_index(drop=True)
    st.dataframe(agg, use_container_width=True)
    agg["block"] = agg["block"].astype(int)
    agg = agg.sort_values("block").reset_index(drop=True)
    st.dataframe(agg, use_container_width=True)
    st.caption("sev-5 MLP minus attention drift, per block. Negative values = attention > MLP (none in any block).")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", "DINOv2 ViT-B/14 (86M params)")
        st.metric("Modules probed", 37)
        st.metric("Gate verdict", gate["verdict"].upper())
        st.metric("Kernel", "arindamtripathi/vitb-drift-profile v1 (Kaggle T4, 259.5s)")
    with c2:
        st.markdown("""
        - 1000 CIFAR-10 test images, seed 42, matched-noise rng 1000+sev.
        - low_light + jpeg, severities 1–5; **two forward passes per condition, no labels, no backprop**.
        - **Late-heavy block profile:** low_light 0.179 (b0) → 0.803/0.866/0.859 (b9/b10/b11); jpeg 0.127 → 0.705/0.790/0.785.
        - **MLP sublayers drift more than attention sublayers in all 12 blocks under both corruptions** (sev5 low_light gaps 0.015–0.159, max at b2; jpeg gaps 0.017–0.108, max at b1).
        - Gate NO-GO with the scale-invariance signature replicated.
        """)

    st.subheader("Drift heatmap (drop from clean)")
    fig = _plot_acc_drift(df, "ViT-B/14 — per-module drift (sev 5, low_light)", "drop_from_clean", "drop_from_clean")
    st.pyplot(fig)

    st.subheader("Drift-weighted rank allocation (drift-weighted vs uniform)")
    fig = _plot_seed_grid(pd.DataFrame({
        "arm": ["uniform", "drift", "fullft"],
        "mean_acc": [0.8394, 0.8417, 0.8461],
        "std": [0.0122, 0.0156, 0.0189],
    }))
    st.pyplot(fig)

    st.markdown("""
    - **Headline numbers (sev 5, unbiased CKA drop):** late-heavy block profile.
    - **Sublayer finding invisible to ViT-S whole-block CKA:** MLP sublayers drift more than attention sublayers in all 12 blocks under both corruptions.
    - **Caveat to flag:** the drift CSVs key `layer` by profiled-module row (0–36). The v9 LoRA consumer (`run_lora_simple_colab.py::load_drift_profile`) expects dense 0..11 block indices — v9 must select the 12 `blocks.k` rows and re-index them 0–11 before `--drift-csv` use.
    """)

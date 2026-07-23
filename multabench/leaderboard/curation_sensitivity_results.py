"""Leaderboard tab: curation sensitivity analysis (NeurIPS rebuttal).

Answers three review questions about the text-tabular curation pipeline (Appendix A.3):
(a) sensitivity to the choice of the 5 curation learners, (b) sensitivity to |M| and the
consensus ratio rho ("3/5"), (c) sensitivity to the acceptance threshold delta (0.001).
"""
import matplotlib.pyplot as plt
import streamlit as st

from multabench.leaderboard.analysis.curation_accept import (
    CURATION_MODELS, compute_deltas, load_pool_5model,
)
from multabench.leaderboard.analysis.delta_sweep import borderline_datasets, delta_sweep
from multabench.leaderboard.analysis.model_sensitivity import (
    agreement_matrix, all_subsets, dataset_stability, extended_model_awareness,
    fleiss_kappa, leave_one_out,
)
from multabench.leaderboard.analysis.threshold_grid import rho_sweep_at_k5, size_rho_grid


@st.cache_data
def _load_deltas():
    df = load_pool_5model()
    return compute_deltas(df)


def _line_chart(df, x, y, title, baseline_x=None):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df[x], df[y], marker="o", color="#3A88C8")
    if baseline_x is not None:
        ax.axvline(baseline_x, color="#B85010", linestyle="--", linewidth=1, label="paper default")
        ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def display_curation_sensitivity():
    st.title("🎯 Curation Sensitivity Analysis")
    st.caption("Text-tabular pool (56 candidates, 23 accepted). All analyses are computed "
               "live from `tabstar_corpus/` + `text_source/` + `more_baselines/` CSVs.")
    deltas = _load_deltas()

    # --- (a) Model / learner-pool sensitivity ---------------------------------
    st.header("(a) Curation-learner sensitivity")

    st.subheader("Leave-one-model-out")
    st.caption("Baseline: 23 accepted with all 5 models, rho=3/5. Dropping any single model "
               "and re-running with the remaining 4 (rho=3/5, i.e. 3-of-4).")
    st.dataframe(leave_one_out(deltas), use_container_width=True)

    st.subheader("All 31 model subsets")
    st.caption("Every non-empty subset of the 5 curation models, rho=3/5. Per-dataset "
               "stability = fraction of the 31 subsets that accept it.")
    subsets = all_subsets(deltas)
    st.dataframe(
        subsets.groupby("size")["n_accepted"].agg(["min", "mean", "max"]).round(1),
        use_container_width=True,
    )
    stability = dataset_stability(subsets, deltas)
    st.dataframe(
        stability.style.background_gradient(cmap="RdYlGn", subset=["frac_subsets_accept"], vmin=0, vmax=1),
        use_container_width=True, height=350,
    )

    st.subheader("Pairwise agreement (Cohen's kappa on per-model accept vote)")
    st.caption("Directly tests whether TabPFNv2 and TabPFN-2.5 (same model family) vote as a bloc.")
    agreement = agreement_matrix(deltas)
    st.dataframe(agreement.style.background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)
    tabpfn_kappa = agreement.loc["TabPFNv2", "TabPFN-2.5"]
    st.metric("TabPFNv2 x TabPFN-2.5 kappa", f"{tabpfn_kappa:.3f}",
              help="Compare to Fleiss' kappa across all 5 models below.")
    st.metric("Fleiss' kappa (all 5 models)", f"{fleiss_kappa(deltas):.3f}")

    st.subheader("Extended-model task-awareness generalization")
    st.caption("Fraction of pool datasets with Delta_Awareness > delta, for the 5 curation "
               "models plus up to 7 additional models (TabDPT, TabICLv2, TabFM, TabPFN-v3, "
               "RealMLP, XGBoost, RandomForest) available only for Frozen/TAR (no Delta_Joint "
               "data exists for these on the pool, so this tests task-awareness only, not "
               "the full accept/reject rule).")
    st.dataframe(extended_model_awareness(), use_container_width=True)

    st.divider()

    # --- (b) |M| and rho sensitivity -------------------------------------------
    st.header("(b) |M| and rho (\"3/5\") sensitivity")

    st.subheader("rho sweep, |M|=5 fixed")
    rho_curve = rho_sweep_at_k5(deltas)
    _line_chart(rho_curve, "rho", "n_accepted", "Accepted count vs. rho (all 5 models)", baseline_x=0.6)
    st.dataframe(rho_curve, use_container_width=True)

    st.subheader("|M| x rho grid")
    st.caption("Mean accepted count across all C(5,k) subsets of each size k, at each rho.")
    grid = size_rho_grid(deltas)
    pivot = grid.pivot(index="rho", columns="size", values="n_accepted_mean")
    st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=23), use_container_width=True)

    st.divider()

    # --- (c) delta sensitivity ---------------------------------------------------
    st.header("(c) delta (acceptance threshold) sensitivity")
    st.caption("Fixed M=5, rho=3/5. Paper default: delta=0.001.")
    sweep = delta_sweep(deltas)
    _line_chart(sweep, "delta", "n_accepted", "Accepted count vs. delta", baseline_x=0.001)
    st.dataframe(sweep, use_container_width=True)

    st.subheader("Borderline datasets")
    st.caption("Currently-accepted datasets whose decision flips to rejected somewhere in "
               "delta in [0, 0.01], and the delta value at which they flip.")
    st.dataframe(borderline_datasets(deltas), use_container_width=True)

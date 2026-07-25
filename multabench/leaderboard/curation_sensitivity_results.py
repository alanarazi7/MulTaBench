"""Leaderboard tab: curation sensitivity analysis (NeurIPS rebuttal).

Answers three review questions about the text-tabular curation pipeline (Appendix A.3):
(a) sensitivity to the choice of the 5 curation learners, (b) sensitivity to |M| and the
consensus ratio rho ("3/5"), (c) sensitivity to the acceptance threshold delta (0.001).
"""
import matplotlib.pyplot as plt
import streamlit as st

from multabench.leaderboard.analysis.curation_accept import (
    CURATION_MODELS, compute_deltas, load_pool_5model, load_pool_10model,
)
from multabench.leaderboard.analysis.delta_sweep import borderline_datasets, delta_sweep
from multabench.leaderboard.analysis.model_sensitivity import (
    agreement_matrix, all_10_choose_5_panels, all_subsets, dataset_stability,
    extended_model_awareness, family_swap, fleiss_kappa, leave_one_out,
    pairwise_subset_agreement, real_alternative_panel, real_per_model_accept_rate,
    stability_distribution, subset_agreement_by_size,
)
from multabench.leaderboard.analysis.threshold_grid import rho_sweep_at_k5, size_rho_grid


@st.cache_data
def _load_deltas():
    df = load_pool_5model()
    return compute_deltas(df)


@st.cache_data
def _load_deltas_10model():
    df = load_pool_10model()
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
    st.caption("How contested is each dataset's decision across all 31 subsets:")
    st.dataframe(stability_distribution(stability), use_container_width=True)

    st.subheader("Pairwise agreement across all model-subset combinations")
    st.caption("Jaccard agreement between the accepted sets of every pair of the 31 model "
               "subsets (C(31,2)=465 pairs) -- generalizes the per-model kappa below to the "
               "full combinatorial space of curation panels, rather than anchoring everything "
               "to the single all-5 baseline.")
    pairwise = pairwise_subset_agreement(subsets)
    col1, col2 = st.columns(2)
    col1.metric("Mean Jaccard, all 465 pairs", f"{pairwise['jaccard'].mean():.3f}")
    majority_only = pairwise[(pairwise["size_1"] >= 3) & (pairwise["size_2"] >= 3)]
    col2.metric("Mean Jaccard, size>=3 pairs only", f"{majority_only['jaccard'].mean():.3f}")
    st.caption("Mean Jaccard by (subset size, subset size):")
    st.dataframe(subset_agreement_by_size(pairwise).style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                 use_container_width=True)

    st.subheader("Pairwise agreement (Cohen's kappa on per-model accept vote)")
    st.caption("Directly tests whether TabPFNv2 and TabPFN-2.5 (same model family) vote as a bloc.")
    agreement = agreement_matrix(deltas)
    st.dataframe(agreement.style.background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)
    tabpfn_kappa = agreement.loc["TabPFNv2", "TabPFN-2.5"]
    st.metric("TabPFNv2 x TabPFN-2.5 kappa", f"{tabpfn_kappa:.3f}",
              help="Compare to Fleiss' kappa across all 5 models below.")
    st.metric("Fleiss' kappa (all 5 models)", f"{fleiss_kappa(deltas):.3f}")

    st.subheader("TabPFN family swap")
    st.caption("A stronger test of the same-family bloc-vote concern than single "
               "leave-one-out: drop BOTH TabPFN variants at once (3 models remain), and the "
               "mirror case of keeping ONLY the TabPFN family (2 models).")
    st.dataframe(family_swap(deltas), use_container_width=True)

    st.divider()
    st.subheader("Full 10-model analysis (5 curation + 5 paper baselines)")
    st.caption("`results/sensitivity/` added the missing no_text/text_only runs for "
               "RandomForest, RealMLP, TabDPT, TabICLv2, XGBoost, so Delta_Joint (and the "
               "REAL Accept(D) rule) is now computable for them too -- everything below is "
               "the real rule, not a Delta_Awareness-only proxy.")
    deltas_10 = _load_deltas_10model()

    st.markdown("**Real per-model accept vote (both conditions), all 10 models**")
    st.dataframe(real_per_model_accept_rate(deltas_10), use_container_width=True)

    st.markdown("**Extended-model task-awareness generalization** (Delta_Awareness alone, "
                 "decoupled from joint signal)")
    st.dataframe(extended_model_awareness(deltas_10), use_container_width=True)

    st.markdown("**Original 5 vs. a fully different 5-model panel** -- directly answers "
                 "\"what if we took any other five\"")
    st.dataframe(real_alternative_panel(deltas_10), use_container_width=True)

    st.markdown("**Every possible 5-model panel drawn from the 10 available models** "
                 "(C(10,5)=252) -- the full combinatorial answer, not just one alternative")
    all_panels = all_10_choose_5_panels(deltas_10)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accepted count", f"{all_panels['n_accepted'].mean():.1f} avg",
              help=f"range: {all_panels['n_accepted'].min()}-{all_panels['n_accepted'].max()}, baseline: 23")
    c2.metric("Jaccard vs. baseline", f"{all_panels['jaccard_vs_baseline'].mean():.3f} avg",
              help=f"range: {all_panels['jaccard_vs_baseline'].min():.3f}-{all_panels['jaccard_vs_baseline'].max():.3f}")
    c3.metric("Panels evaluated", len(all_panels))
    st.caption("Mean by how many of the panel's 5 models are original curation models:")
    by_n_curation = all_panels.groupby("n_curation_models")[["n_accepted", "jaccard_vs_baseline"]].mean().round(3)
    st.dataframe(by_n_curation, use_container_width=True)
    with st.expander("All 252 panels"):
        st.dataframe(all_panels.sort_values("jaccard_vs_baseline"), use_container_width=True, height=400)

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

"""Selects the 20 additional text-tabular datasets that take MulTaBench-Full from 20 to 40.

MulTaBench-Core admits a dataset only if Joint Signal AND Tabular Awareness both hold
(pass_matrix.passes). MulTaBench-Full relaxes this to **Joint Signal only**
(pass_matrix.joint_signal_passes): the joint frozen model must beat both unimodal models, which
is exactly the condition that rejects pure-NLP tasks (the text alone is as good) and
redundant-text tasks (the tabular features alone are). Whether fine-tuning the encoder gives an
additional gain is not required for Full.

Same committee and thresholds as the paper: delta = 0.001, rho = 3/5 over the 5 curation models
(LightGBM, CatBoost, TabM, TabPFNv2, TabPFN-2.5). The 5 extra models are reported alongside as a
robustness signal (joint_pass_10) but do not decide admission.

Selection rule, applied to the non-Core datasets of the 56-dataset pool that pass Joint Signal:
  1. sort by (joint_pass_5, joint_pass_10, median_delta_joint), all descending;
  2. take the top 20.
Unanimous 5/5 passers therefore all get in before any 4/5 candidate is considered. The script
also prints a domain histogram and the domain overlap with Core, so the tail of the ranking can
be adjusted by hand for diversity -- swaps are one-line edits in multabench/datasets/tiers.py.

Reads committee_pool.py's pool_scores_long.csv. Writes text_full_selection.csv.

Run standalone: `python -m multabench.leaderboard.analysis.text_full_selection`
"""
from collections import Counter
from os.path import dirname, join

import pandas as pd

from multabench.datasets.text_benchmarks import (
    AUTOML_MULTIMODAL,
    CARTE_BENCHMARK,
    TEXT_TAB_BENCH,
    VECTORIZING,
)
from multabench.datasets.utils import dataset_from_name
from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, EXTRA_MODELS
from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, compute_deltas
from multabench.leaderboard.main_paper.text_pool import _MULTABENCH_POOL_NAMES

_RESULTS = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")
_SCORES_CSV = join(_RESULTS, "pool_scores_long.csv")
_OUT_CSV = join(_RESULTS, "text_full_selection.csv")

RHO = 3 / 5
N_TEXT_FULL_EXTRA = 20

_SOURCE_PAPERS = {
    "AutoMLMultimodal": AUTOML_MULTIMODAL,
    "Vectorizing": VECTORIZING,
    "CARTE": CARTE_BENCHMARK,
    "TextTabBench": TEXT_TAB_BENCH,
}


def _source_paper_by_name() -> dict:
    """Which source paper each candidate came from. A dataset can appear in more than one."""
    papers = {}
    for paper, datasets in _SOURCE_PAPERS.items():
        for dataset_id in datasets:
            papers.setdefault(dataset_id.name, []).append(paper)
    return {name: "+".join(sorted(paper)) for name, paper in papers.items()}


def _domain(dataset: str) -> str:
    """The domain token of the naming convention: {TASK}_{MODALITY}_{DOMAIN}_{NAME}."""
    parts = dataset.split("_")
    return parts[2] if len(parts) > 2 else "?"


def build_selection_table(df: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """One row per pool dataset, with per-committee Joint-Signal counts and the Full decision."""
    deltas = {}
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        deltas.setdefault(dataset, {})[model] = compute_deltas(sub)

    papers = _source_paper_by_name()
    quorum = RHO * len(CURATION_MODELS)
    rows = []
    for dataset, per_model in deltas.items():
        curation = [per_model[m] for m in CURATION_MODELS if m in per_model]
        everyone = [per_model[m] for m in CURATION_MODELS + EXTRA_MODELS if m in per_model]
        joint_curation = pd.Series([d[0] for d in curation])
        joint_pass_5 = int((joint_curation > delta).sum())
        rows.append({
            "dataset": dataset,
            "source_paper": papers.get(dataset, "-"),
            "task_type": dataset.split("_")[0],
            "domain": _domain(dataset),
            "in_core": dataset in _MULTABENCH_POOL_NAMES,
            "n_curation_models": len(curation),
            "n_all_models": len(everyone),
            "joint_pass_5": joint_pass_5,
            "joint_pass_10": int(sum(1 for d in everyone if d[0] > delta)),
            "awareness_pass_5": int(sum(1 for d in curation if d[1] > delta)),
            "core_pass_5": int(sum(1 for d in curation if d[0] > delta and d[1] > delta)),
            "median_delta_joint": round(joint_curation.median(), 4),
            "min_delta_joint": round(joint_curation.min(), 4),
            # Ratio-scaled quorum, matching committee_panel_pass_rates.accept_at_quorum: a
            # dataset no curation model can be evaluated on must not pass on an empty vote.
            "joint_accept": bool(len(curation) > 0 and joint_pass_5 >= RHO * len(curation)),
            "core_decision": "accept" if sum(
                1 for d in curation if d[0] > delta and d[1] > delta
            ) >= quorum else "reject",
        })

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["joint_pass_5", "joint_pass_10", "median_delta_joint"], ascending=False
    ).reset_index(drop=True)

    candidates = table[(~table["in_core"]) & table["joint_accept"]]
    selected = list(candidates["dataset"])[:N_TEXT_FULL_EXTRA]
    ranks = {dataset: i + 1 for i, dataset in enumerate(selected)}
    table["selected_full"] = table["dataset"].isin(selected)
    table["rank"] = table["dataset"].map(ranks)
    table["exclusion_reason"] = [
        "" if row.selected_full
        else "already in MulTaBench-Core" if row.in_core
        else f"fails Joint Signal ({row.joint_pass_5}/{row.n_curation_models} models)" if not row.joint_accept
        else f"ranked below the top {N_TEXT_FULL_EXTRA} (Joint Signal {row.joint_pass_5}/{row.n_curation_models})"
        for row in table.itertuples()
    ]
    return table


def _print_report(table: pd.DataFrame) -> None:
    n_core = int(table["in_core"].sum())
    print(f"\nPool: {len(table)} datasets ({n_core} in Core, {len(table) - n_core} not)")
    print(f"Joint Signal accepts (delta={DELTA_DEFAULT}, rho=3/5): {int(table['joint_accept'].sum())}")
    print(f"Core rule accepts (Joint AND Awareness):               "
          f"{int((table['core_decision'] == 'accept').sum())}")

    candidates = table[(~table["in_core"]) & table["joint_accept"]]
    print(f"\nNon-Core Joint-Signal passers: {len(candidates)} — selecting {N_TEXT_FULL_EXTRA}")
    cols = ["rank", "dataset", "domain", "joint_pass_5", "joint_pass_10", "awareness_pass_5",
            "median_delta_joint", "selected_full", "source_paper"]
    print(candidates[cols].to_string(index=False))

    selected = table[table["selected_full"]]
    print("\nDomain histogram of the 20 selected:")
    for domain, n in Counter(selected["domain"]).most_common():
        in_core = sum(1 for d in table[table["in_core"]]["domain"] if d == domain)
        print(f"  {domain:<16} {n:>2}   (Core already has {in_core})")
    print("\nTask types of the 20 selected: " + ", ".join(
        f"{task} {n}" for task, n in Counter(selected["task_type"]).most_common()))

    print("\nRegistry snippet for MULTABENCH_FULL_TEXT_EXTRA in multabench/datasets/tiers.py:")
    for row in selected.sort_values("rank").itertuples():
        dataset_id = dataset_from_name(row.dataset)
        print(f"    {type(dataset_id).__name__}.{dataset_id.name},"
              f"  # joint {row.joint_pass_5}/5, {row.joint_pass_10}/10 — {row.source_paper}")


def main() -> None:
    df = pd.read_csv(_SCORES_CSV)
    table = build_selection_table(df)
    table.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {len(table)} rows to {_OUT_CSV}")
    _print_report(table)


if __name__ == "__main__":
    main()

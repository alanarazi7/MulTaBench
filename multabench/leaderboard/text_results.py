from os.path import dirname, join

import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import ALL_FEAT, FINETUNED, MODE, NO_TEXT, TEXT_ONLY, FOLD, MODEL, DATASET, TEST_SCORE, E5_TUNE, MM, E5_MODEL, E5_SMALL, E5_LARGE, E5_SMALL_ALL, E5_SMALL_FT, E5_LARGE_ALL, E5_LARGE_FT, IS_TUNED, DINO_MODEL, DINO_SMALL
from multabench.leaderboard.plots import plot_dataset_performance
from multabench.leaderboard.utils import badge
from multabench.datasets.text_benchmarks import ACCEPTED_TEXT_DATASETS, REJECTED_TEXT_DATASETS

CONDITIONS = [TEXT_ONLY, NO_TEXT, ALL_FEAT, FINETUNED]
E5_CONDITIONS = [E5_SMALL_ALL, E5_SMALL_FT, E5_LARGE_ALL, E5_LARGE_FT]

_APPROVED_NAMES = {d.name for d in ACCEPTED_TEXT_DATASETS}
_REJECTED_NAMES = {d.name for d in REJECTED_TEXT_DATASETS}

_CORPUS_CSV = join(dirname(__file__), 'results', 'tabstar_corpus', 'text_50_datasets.csv')
_TEXTTABENCH_CSV = join(dirname(__file__), 'results', 'tabstar_corpus', 'texttabench_datasets.csv')


@st.cache_data
def _load_pool_corpus_data() -> pd.DataFrame:
    df = pd.read_csv(_CORPUS_CSV)
    df2 = pd.read_csv(_TEXTTABENCH_CSV)
    if "dataset_name" in df2.columns and DATASET not in df2.columns:
        df2 = df2.rename(columns={"dataset_name": DATASET})
    df = pd.concat([df, df2], ignore_index=True)
    df = df.dropna(subset=[MODEL])
    df[MODEL] = df[MODEL].apply(lambda x: x.strip())
    df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
    if E5_MODEL not in df.columns:
        df[E5_MODEL] = E5_SMALL
    if DINO_MODEL not in df.columns:
        df[DINO_MODEL] = DINO_SMALL
    if TEST_SCORE in df.columns:
        df[TEST_SCORE] = df[TEST_SCORE].clip(lower=-0.1)
    return df


_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]
_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}


def compute_curation_grid(df: pd.DataFrame) -> pd.DataFrame:
    from multabench.leaderboard.main_paper.text_pool import _MULTABENCH_POOL_NAMES as _MB_NAMES
    """Return per-dataset × per-model all_three pass/fail grid.

    Rows: datasets (56 candidates), sorted approved-first then by all_three count desc.
    Columns: LightGBM, CatBoost, TabM, TabPFNv2, TabPFN-2.5, then all_three (/5), decision, reason.
    Values: True/False for each model column.
    """
    raw = df.copy()
    raw[MM] = raw[MODE].map({ALL_FEAT: "all", FINETUNED: "ft", TEXT_ONLY: "text_only", NO_TEXT: "no_text"})
    raw["model_label"] = raw[MODEL].map(_MODEL_LABELS)
    raw = raw[raw["model_label"].notna()]

    avg = raw.groupby([DATASET, "model_label", MM])[TEST_SCORE].mean().reset_index()
    avg[TEST_SCORE] = avg[TEST_SCORE].round(3)
    pivot = avg.pivot_table(index=[DATASET, "model_label"], columns=MM, values=TEST_SCORE).reset_index()
    pivot["all_three"] = (
        (pivot["ft"] > pivot["all"]) &
        (pivot["all"] > pivot["no_text"]) &
        (pivot["all"] > pivot["text_only"])
    )

    grid = pivot.pivot_table(index=DATASET, columns="model_label", values="all_three")
    for m in _MODEL_ORDER:
        if m not in grid.columns:
            grid[m] = False
    grid = grid[_MODEL_ORDER]

    grid["all_three (/5)"] = grid[_MODEL_ORDER].sum(axis=1).astype(int)
    grid["decision"] = grid.index.map(
        lambda d: "approved" if (d in _APPROVED_NAMES or d in _MB_NAMES) else ("rejected" if d in _REJECTED_NAMES else "unknown")
    )
    grid["reason"] = grid.apply(_rejection_reason, axis=1)
    return grid.sort_values(["decision", "all_three (/5)"], ascending=[True, False])


def display_pool_performance():
    st.title("🏊 Pool Performance")
    df = _load_pool_corpus_data()
    df[MODE] = df.apply(single_variant_name, axis=1)

    summary = _compute_curation_summary(df)
    _display_per_dataset_curation(df, summary)
    with st.expander("Dataset Summary"):
        st.dataframe(summary)


def _display_e5_comparison(df: pd.DataFrame):
    st.subheader("E5 Model Comparison — Benchmark Datasets")
    st.caption("Comparing e5-small (all), e5-small (ft), and e5-large (all) on approved benchmark datasets")

    # Keep only approved benchmark datasets
    df = df[df[DATASET].isin(_APPROVED_NAMES)].copy()

    # Build e5_mode label
    df["e5_mode"] = df.apply(_e5_mode_label, axis=1)
    df = df[df["e5_mode"].notna()]

    for dataset in sorted(df[DATASET].unique()):
        st.markdown(f"### {dataset}")
        df_ds = df[df[DATASET] == dataset]
        plot_df = df_ds.groupby([MODEL, "e5_mode"]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_df = plot_df.rename(columns={"e5_mode": MODE})
        plot_dataset_performance(plot_df, conditions=E5_CONDITIONS, title="")
        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, "e5_mode"], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()


def _e5_mode_label(row) -> str | None:
    mm = row[MM]
    e5 = row.get(E5_MODEL, E5_SMALL)
    if e5 == E5_LARGE and mm == "all":
        return E5_LARGE_ALL
    if e5 == E5_LARGE and mm == "ft":
        return E5_LARGE_FT
    if e5 == E5_SMALL and mm == "all":
        return E5_SMALL_ALL
    if e5 == E5_SMALL and mm == "ft":
        return E5_SMALL_FT
    return None


_REASON_LABELS = {
    "[1]":   "text ≥ all — no useful tabular features",
    "[2]":   "no_text ≥ all — text is irrelevant",
    "[3]":   "all ≥ ft — fine-tuning adds no value",
    "[3*]":  "inconsistent signal across models (all_three < 3/5)",
    "[1+2]": "text ≥ all and no_text ≥ all",
}


def _rejection_reason(row) -> str:
    def _r(key):
        v = row.get(key, float("nan"))
        return round(float(v), 3) if pd.notna(v) else float("nan")
    text, no_text, all_, ft = _r("text_only"), _r("no_text"), _r("all"), _r("ft")
    r1 = pd.notna(text)    and pd.notna(all_) and text    >= all_
    r2 = pd.notna(no_text) and pd.notna(all_) and no_text >= all_
    r3 = pd.notna(all_)    and pd.notna(ft)   and all_    >= ft
    if r1 and r2:   return "[1+2]"
    elif r1:        return "[1]"
    elif r2:        return "[2]"
    elif r3:        return "[3]"
    else:           return "[3*]"


def _compute_curation_summary(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    raw[MM] = raw[MODE].map({ALL_FEAT: "all", FINETUNED: "ft", TEXT_ONLY: "text_only", NO_TEXT: "no_text"})
    avg = raw.groupby([DATASET, MODEL, MM])[TEST_SCORE].mean().reset_index()
    avg[TEST_SCORE] = avg[TEST_SCORE].round(3)
    pivot = avg.pivot_table(index=[DATASET, MODEL], columns=MM, values=TEST_SCORE).reset_index()
    pivot["ft_beats_all"] = pivot["ft"] > pivot["all"]
    pivot["all_beats_no_text"] = pivot["all"] > pivot["no_text"]
    pivot["all_beats_text_only"] = pivot["all"] > pivot["text_only"]
    pivot["all_three"] = pivot["ft_beats_all"] & pivot["all_beats_no_text"] & pivot["all_beats_text_only"]

    counts = pivot.groupby(DATASET)[["ft_beats_all", "all_beats_no_text", "all_beats_text_only", "all_three"]].sum().astype(int)
    counts.columns = ["ft>all (/5)", "all>no-text (/5)", "all>text-only (/5)", "all_three (/5)"]

    means = avg.pivot_table(index=DATASET, columns=MM, values=TEST_SCORE, aggfunc="mean")[
        ["no_text", "text_only", "all", "ft"]
    ].round(3)

    summary = counts.join(means)
    summary["reason"] = summary.apply(_rejection_reason, axis=1)
    summary["decision"] = summary.index.map(
        lambda d: "✅ approved" if d in _APPROVED_NAMES else ("❌ rejected" if d in _REJECTED_NAMES else "❓ unknown")
    )
    return summary.sort_values(["decision", "ft>all (/5)"], ascending=[True, False])



def _display_per_dataset_curation(df: pd.DataFrame, summary: pd.DataFrame):
    for dataset, row in summary.iterrows():
        approved = dataset in _APPROVED_NAMES
        rejected = dataset in _REJECTED_NAMES
        if not (approved or rejected):
            st.warning(f"{dataset} is neither approved nor rejected")
        approval_icon = "✅" if approved else ("❌" if rejected else "❓")
        st.markdown(f"### {approval_icon} {dataset}")
        ft_all = int(row['ft>all (/5)'])
        no_txt = int(row['all>no-text (/5)'])
        txt    = int(row['all>text-only (/5)'])
        all_three = int(row['all_three (/5)'])
        st.caption(f"{badge(ft_all)} ft>all: {ft_all}/5  |  {badge(no_txt)} all>no-text: {no_txt}/5  |  {badge(txt)} all>text-only: {txt}/5  |  {badge(all_three)} all three: {all_three}/5")
        passes = all_three >= 3
        reason = row.get("reason", "")
        if approved and not passes:
            st.error(f"INCONSISTENCY: {dataset} is APPROVED but only {all_three}/5 models satisfy all three conditions")
        elif rejected and passes:
            st.error(f"INCONSISTENCY: {dataset} is REJECTED but {all_three}/5 models satisfy all three conditions")
        elif rejected:
            st.info(f"Rejected: {reason} — {_REASON_LABELS.get(reason, '')}")
        df_ds = df[df[DATASET] == dataset]
        plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: 'mean'}).reset_index()
        plot_dataset_performance(plot_df, conditions=CONDITIONS, title="")
        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()


def single_variant_name(row):
    mapping = {
        "all": ALL_FEAT,
        "ft": FINETUNED,
        "text_only": TEXT_ONLY,
        "no_text": NO_TEXT,
    }
    return mapping[row[MM]]

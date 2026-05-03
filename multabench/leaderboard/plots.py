import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import streamlit as st
from pandas import DataFrame

from multabench.leaderboard.data.keys import MODEL, TEST_SCORE, MODE, DATASET, FOLD, ALL_FEAT, FINETUNED

COLORS = ['#FFCC99', '#FFD1DC', '#8DE5A1', '#A1C9F4', '#D4B8E0', '#FFE4B5', '#B0E0E6', '#F4A896']

_COLOR_FROZEN    = 'paleturquoise'
_COLOR_FINETUNED = '#A9CCE3'


def render_filters(df: DataFrame, key: str) -> DataFrame:
    models = sorted(df[MODEL].apply(lambda x: x.split()[0]).unique())
    modes  = sorted(df[MODE].unique())
    c1, c2 = st.columns(2)
    with c1:
        sel_models = st.multiselect("Models", models, default=models, key=f"{key}_models")
    with c2:
        sel_modes = st.multiselect("Modes", modes, default=modes, key=f"{key}_modes")
    df = df[df[MODEL].apply(lambda x: x.split()[0]).isin(sel_models)]
    df = df[df[MODE].isin(sel_modes)]
    return df


def plot_aggregated_performance(df: DataFrame, title: str):
    from matplotlib.patches import Patch

    df = df.copy()
    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    df[TEST_SCORE] = df[TEST_SCORE] * 100
    df = df[df[MODE].isin([ALL_FEAT, FINETUNED])]

    agg = (df.groupby([MODEL, MODE])[TEST_SCORE]
             .agg(mean='mean', std='std', n='count')
             .reset_index())
    agg['ci'] = 1.96 * agg['std'] / agg['n'] ** 0.5

    # Show numbers so we can verify them
    pivot = agg.pivot(index=MODEL, columns=MODE, values='mean').round(2)
    st.dataframe(pivot)

    # Sort by ft mean descending (best at top → highest y index)
    ft_mean = agg[agg[MODE] == FINETUNED].set_index(MODEL)['mean']
    models  = ft_mean.sort_values(ascending=False).index.tolist()
    n = len(models)
    # y centres: model 0 = top = n-1, model n-1 = bottom = 0
    y_of = {m: n - 1 - i for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7, n * 1.0 + 0.8))

    bar_h = 0.25
    dy    = 0.18   # offset from row centre to each sub-bar centre

    for mode, offset, color in [(ALL_FEAT, +dy, _COLOR_FROZEN),
                                 (FINETUNED, -dy, _COLOR_FINETUNED)]:
        subset = agg[agg[MODE] == mode].set_index(MODEL)
        for model in models:
            if model not in subset.index:
                continue
            row  = subset.loc[model]
            y    = y_of[model] + offset
            ax.barh(y, row['mean'], height=bar_h, color=color, linewidth=0)
            ax.errorbar(row['mean'], y, xerr=row['ci'],
                        fmt='none', ecolor='black', capsize=3, linewidth=0.8)

    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels(list(y_of.keys()), fontsize=11)
    ax.set_xlabel("Test Score (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color='0.9', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=_COLOR_FROZEN,    label='Frozen (all)'),
                       Patch(color=_COLOR_FINETUNED, label='Fine-tuned (ft)')],
              loc='lower right', frameon=False, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _normalize_scores(df: DataFrame, group_cols: list[str]) -> DataFrame:
    lo = df.groupby(group_cols)[TEST_SCORE].transform("min")
    hi = df.groupby(group_cols)[TEST_SCORE].transform("max")
    span = (hi - lo).clip(lower=1e-9)
    df = df.copy()
    df["norm_score"] = (df[TEST_SCORE] - lo) / span
    return df


def plot_normalized_overview(df: DataFrame, conditions: list[str], key: str):
    within_model = st.checkbox("Normalize within model", key=f"norm_{key}")
    group_cols = [DATASET, FOLD, MODEL] if within_model else [DATASET, FOLD]
    caption = ("Within each (dataset, fold, model), conditions rescaled to [0, 1]." if within_model
               else "Within each (dataset, fold), all models × conditions rescaled to [0, 1].")
    st.caption(caption + " Mean ± 95% CI.")

    df = _normalize_scores(df, group_cols)

    agg = (df.groupby([MODEL, MODE])["norm_score"]
           .agg(mean="mean", std="std", n="count")
           .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    agg[MODEL] = agg[MODEL].apply(lambda x: x.split()[0])

    models = sorted(agg[MODEL].unique())
    x = np.arange(len(models))
    width = 0.18
    offset = (len(conditions) - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cond in enumerate(conditions):
        subset = agg[agg[MODE] == cond].set_index(MODEL).reindex(models).reset_index()
        bars = ax.bar(x + (i - offset) * width, subset["mean"], width,
                      label=cond, color=COLORS[i % len(COLORS)],
                      edgecolor="white", linewidth=1.2)
        ax.errorbar(x + (i - offset) * width, subset["mean"], yerr=subset["ci"],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=8, fontweight="bold", color="#444")

    ax.set_ylabel("Normalized score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Mode", bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


_COLOR_END_TO_END = '#D4AF37'   # gold — salient, distinct from the blue/teal palette


def plot_ft_vs_all_normalized(df: DataFrame, native_models: set[str] | None = None):
    """Aggregate overview: ft vs all, normalized within each (dataset, fold) across all model×state combos."""
    df = df[df[MODE].isin([ALL_FEAT, FINETUNED])].copy()
    if df.empty:
        st.info("Not enough ft/all data yet.")
        return

    df = _normalize_scores(df, group_cols=[DATASET, FOLD])

    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    agg = (df.groupby([MODEL, MODE])["norm_score"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5

    all_means = agg[agg[MODE] == ALL_FEAT].set_index(MODEL)["mean"]
    all_models = agg[MODEL].unique()
    models = sorted(all_models, key=lambda m: all_means.get(m, float("-inf")))
    x = np.arange(len(models))
    width = 0.32
    offset = 0.5 * width

    native = native_models or set()

    fig, ax = plt.subplots(figsize=(10, 4))
    for mode, dx, color_std, label_std in [
        (ALL_FEAT,  -offset, _COLOR_FROZEN,    "Frozen (all)"),
        (FINETUNED, +offset, _COLOR_FINETUNED, "Fine-tuned (ft)"),
    ]:
        subset = agg[agg[MODE] == mode].set_index(MODEL).reindex(models).reset_index()
        # end-to-end models only exist in ALL_FEAT; use their special color there, standard elsewhere
        colors = [_COLOR_END_TO_END if (m in native and mode == ALL_FEAT) else color_std for m in models]
        bars = ax.bar(x + dx, subset["mean"], width, color=colors, edgecolor="white", linewidth=1.2)
        ax.errorbar(x + dx, subset["mean"], yerr=subset["ci"],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=9, fontweight="bold", color="#444")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_COLOR_FROZEN,    label="Frozen (all)"),
        Patch(color=_COLOR_FINETUNED, label="Fine-tuned (ft)"),
    ]
    if native:
        legend_handles.append(Patch(color=_COLOR_END_TO_END, label="End-to-end"))

    ax.set_ylabel("Normalized score (0–1)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("ft vs all — normalized per fold across all models × states", fontsize=12, fontweight="bold")
    ax.legend(handles=legend_handles, frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_dataset_performance(df: DataFrame, conditions: list[str], title: str, bar_width: float = 0.18):
    df = df.copy()
    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    df[TEST_SCORE] = df[TEST_SCORE] * 100
    models = sorted(df[MODEL].unique())

    x = np.arange(len(models))
    width = bar_width
    offset = (len(conditions) - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cond in enumerate(conditions):
        subset = df[df[MODE] == cond].set_index(MODEL).reindex(models).reset_index()
        rects = ax.bar(x + (i - offset) * width, subset[TEST_SCORE], width,
                       label=cond, color=COLORS[i % len(COLORS)],
                       edgecolor='white', linewidth=1.2)
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=9, fontweight='bold', color='#444')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    ax.set_ylabel("Test Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha='right', fontsize=11)
    filled = df[TEST_SCORE].dropna()
    if len(filled):
        ax.set_ylim(max(filled.min() - 10, 0), min(filled.max() + 10, 102))
    ax.legend(title="Mode", bbox_to_anchor=(1, 1), loc='upper left', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

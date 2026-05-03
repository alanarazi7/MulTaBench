import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Patch

from multabench.leaderboard.data.keys import DATASET, FOLD, MODEL, MODE, MM, TEST_SCORE
from multabench.leaderboard.data.loading import load_triple_data

# Local display labels for this panel
_IMG        = "Img"
_TAB_IMG    = "Tab+Img"
_TXT        = "Txt"
_TAB_TXT    = "Tab+Txt"
_ALL        = "All"
_ALL_FT_TXT = "All+FT Txt"
_ALL_FT_IMG = "All+FT Img"
_ALL_2X_FT  = "All+2x FT"

_TRIPLE_MODE_MAP = {
    "img":           _IMG,
    "non_txt":       _TAB_IMG,
    "txt":           _TXT,
    "non":           _TAB_TXT,
    "all":           _ALL,
    "ft-txt":        _ALL_FT_TXT,
    "ft":            _ALL_FT_IMG,
    "ft-img-ft-txt": _ALL_2X_FT,
}

# Progression: unimodal → add tabular → add all frozen → add finetuning
_TRIPLE_CONDITIONS = [_IMG, _TAB_IMG, _TXT, _TAB_TXT, _ALL, _ALL_FT_TXT, _ALL_FT_IMG, _ALL_2X_FT]

# (facecolor, hatch, edgecolor)
# Blue = image family, green = text family, gray = neutral, orange = best
# Solid = frozen, hatched = includes finetuning
_STYLE = {
    _IMG:        ('#A1C9F4', '',     'white'),
    _TAB_IMG:    ('#4472C4', '',     'white'),
    _TXT:        ('#8DE5A1', '',     'white'),
    _TAB_TXT:    ('#2E8B57', '',     'white'),
    _ALL:        ('#B8B8B8', '',     'white'),
    _ALL_FT_TXT: ('#2E8B57', '//',   '#1A5233'),
    _ALL_FT_IMG: ('#4472C4', '\\\\', '#1F3060'),
    _ALL_2X_FT:  ('#E8924A', 'xx',   '#7A3A10'),
}

_EXPECTED_FOLDS = set(range(5))


def display_triple():
    st.title("😍 True Multimodality")

    df = load_triple_data()
    if df.empty:
        st.warning("No Triple data found.")
        return

    df[MODE] = df[MM].map(_TRIPLE_MODE_MAP)
    df = df[df[MODE].notna()]

    for dataset in sorted(df[DATASET].unique()):
        df_ds = df[df[DATASET] == dataset]
        if not set(_TRIPLE_CONDITIONS).issubset(set(df_ds[MODE].unique())):
            continue

        st.markdown(f"### {dataset}")

        missing_folds = _EXPECTED_FOLDS - set(df_ds[FOLD].astype(int).unique())
        if missing_folds:
            st.warning(f"Missing folds: {sorted(missing_folds)}")

        plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
        _plot_triple(plot_df)
        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()


def _plot_triple(df):
    df = df.copy()
    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    df[TEST_SCORE] = df[TEST_SCORE] * 100
    models = sorted(df[MODEL].unique())

    n = len(_TRIPLE_CONDITIONS)
    x = np.arange(len(models))
    width = 0.10
    offset = (n - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cond in enumerate(_TRIPLE_CONDITIONS):
        fc, hatch, ec = _STYLE[cond]
        subset = df[df[MODE] == cond].set_index(MODEL).reindex(models).reset_index()
        rects = ax.bar(
            x + (i - offset) * width,
            subset[TEST_SCORE],
            width,
            facecolor=fc,
            hatch=hatch,
            edgecolor=ec,
            linewidth=0.8,
            label=cond,
        )
        # Labels are above the bars on white background — always use dark color
        ax.bar_label(rects, padding=2, fmt='%.1f', fontsize=8, fontweight='bold', color='#333')

    filled = df[TEST_SCORE].dropna()
    if len(filled):
        ax.set_ylim(max(filled.min() - 10, 0), min(filled.max() + 18, 108))
    ax.set_ylabel("Test Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha='right', fontsize=11)

    legend_handles = [
        Patch(facecolor=_STYLE[c][0], hatch=_STYLE[c][1], edgecolor=_STYLE[c][2], label=c)
        for c in _TRIPLE_CONDITIONS
    ]
    ax.legend(handles=legend_handles, title="Mode", bbox_to_anchor=(1, 1),
              loc='upper left', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

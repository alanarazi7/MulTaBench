"""Paper figure: PetFinder trimodal case study (§6.2).

All 8 modality conditions across five learners, showing the independent
contribution of structured features, image embeddings, and text embeddings:

  txt           — text embeddings only
  img           — image embeddings only
  non           — structured features only
  non_txt       — structured + frozen image
  all           — structured + frozen image + frozen text
  ft            — structured + fine-tuned image
  ft-txt        — structured + fine-tuned text
  ft-img-ft-txt — structured + fine-tuned image + fine-tuned text
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CSV_IMG = os.path.join(_HERE, "..", "leaderboard", "results", "images", "MUL_IMAGE_PETFINDER.csv")
_CSV_TRI = os.path.join(_HERE, "..", "leaderboard", "results", "tabular_image_text", "MUL_IMAGE_PETFINDER.csv")
_OUT   = os.path.join(_HERE, "../../../paper-multabench/figures/petfinder.pdf")

_BAR_EDGE = "#555"

# ---------------------------------------------------------------------------
# Conditions — image family first, then text family, then combined
# Mapping from triple_results.py: non=Tab+Txt, non_txt=Tab+Img, ft=All+FT Img, ft-txt=All+FT Txt
# ---------------------------------------------------------------------------
_STATES = ["img", "non_txt", "txt", "non", "all", "ft-txt", "ft", "ft-img-ft-txt"]

_LABELS = {
    "img":           "Image",
    "non_txt":       "Struct. + Image",
    "txt":           "Text",
    "non":           "Struct. + Text",
    "all":           "Struct. + Image + Text",
    "ft-txt":        "Struct. + Image + Text FT",
    "ft":            "Struct. + Image FT + Text",
    "ft-img-ft-txt": "Struct. + Image FT + Text FT",
}

# Blue = image family, green = text family, gray = all-frozen combined, orange = best FT
# Solid = frozen encoders; hatched = fine-tuned on target task
_COLORS = {
    "img":           "#A1C9F4",   # light blue    — image only
    "non_txt":       "#4472C4",   # medium blue   — struct + frozen image
    "txt":           "#8DE5A1",   # light green   — text only
    "non":           "#2E8B57",   # dark green    — struct + frozen text
    "all":           "#B8B8B8",   # gray          — struct + both frozen
    "ft-txt":        "#2E8B57",   # dark green    — struct + image + text FT
    "ft":            "#4472C4",   # medium blue   — struct + image FT + text
    "ft-img-ft-txt": "#E8924A",   # orange        — struct + both FT (best)
}

# Solid = frozen; hatched = fine-tuned
_HATCHES = {
    "img":           "",
    "non_txt":       "",
    "txt":           "",
    "non":           "",
    "all":           "",
    "ft-txt":        "//",
    "ft":            "\\\\",
    "ft-img-ft-txt": "xx",
}

_EDGECOLORS = {
    "img":           _BAR_EDGE,
    "non_txt":       _BAR_EDGE,
    "txt":           _BAR_EDGE,
    "non":           _BAR_EDGE,
    "all":           _BAR_EDGE,
    "ft-txt":        "#1A5233",
    "ft":            "#1F3060",
    "ft-img-ft-txt": "#7A3A10",
}

_MODEL_LABELS = {
    "LightGBM 💡":    "LightGBM",
    "CatBoost 😸":    "CatBoost",
    "TabM Ⓜ️":       "TabM",
    "TabPFN-v2 🤯":   "TabPFN-v2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-v2.5",
}
_MODEL_ORDER = list(_MODEL_LABELS.keys())

_CAPSIZE  = 1.5
_ERR_LW   = 0.7
_FS_TITLE  = 14
_FS_TICK   = 11
_FS_LEGEND = 10
_Y_MIN     = 0.68


def _ci95(vals: np.ndarray) -> float:
    return 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0


def main() -> None:
    img = pd.read_csv(_CSV_IMG)
    tri = pd.read_csv(_CSV_TRI)
    img.columns = [c.strip() for c in img.columns]
    tri.columns = [c.strip() for c in tri.columns]
    df = pd.concat([img, tri], ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 4.0))

    n_models = len(_MODEL_ORDER)
    n_states = len(_STATES)
    bar_w    = 0.10
    x_base   = np.arange(n_models) * 1.15  # extra spacing between model groups

    for si, state in enumerate(_STATES):
        offsets = x_base + (si - (n_states - 1) / 2) * bar_w
        means, errs = [], []
        for model in _MODEL_ORDER:
            vals = df[(df["multimodal_state"] == state) & (df["model"] == model)]["test_score"].values
            means.append(vals.mean() if len(vals) else float("nan"))
            errs.append(_ci95(vals) if len(vals) else 0.0)

        ax.bar(
            offsets, means,
            width=bar_w,
            color=_COLORS[state],
            edgecolor=_EDGECOLORS[state],
            linewidth=0.4,
            hatch=_HATCHES[state],
            label=_LABELS[state],
            yerr=errs,
            capsize=_CAPSIZE,
            error_kw={"elinewidth": _ERR_LW, "ecolor": _BAR_EDGE},
            zorder=3,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([_MODEL_LABELS[m] for m in _MODEL_ORDER], fontsize=_FS_TICK)
    ax.set_ylabel("AUROC", fontsize=_FS_TICK)
    ax.set_ylim(_Y_MIN, 0.92)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.set_title("PetFinder", fontsize=_FS_TITLE)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        fontsize=_FS_LEGEND,
        framealpha=0.9,
        edgecolor="#ccc",
        ncol=2,
        title="Blue = image family | Green = text family | Hatched = fine-tuned",
        title_fontsize=7.5,
    )

    fig.tight_layout()
    fig.savefig(_OUT, bbox_inches="tight")
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    main()

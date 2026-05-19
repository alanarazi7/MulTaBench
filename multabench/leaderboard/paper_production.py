"""
Production exports for the MulTaBench paper.
Generates paper-quality figures (PNG/PDF) and LaTeX tables from live data.
"""
import io
from os import listdir
from os.path import dirname, join, expanduser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from multabench.scripts.do_dataset_summary import _SUMMARY_CSV
from multabench.datasets.text_benchmarks import (
    AUTOML_MULTIMODAL_ACCEPTED, AUTOML_MULTIMODAL_REJECTED,
    VECTORIZING_ACCEPTED, VECTORIZING_REJECTED,
    CARTE_ACCEPTED, CARTE_REJECTED,
    TEXT_TAB_BENCH_ACCEPTED, TEXT_TAB_BENCH_REJECTED,
    ACCEPTED_TEXT_DATASETS, REJECTED_TEXT_DATASETS,
)

from multabench.leaderboard.main_paper.curation_example import make_fig        as _make_curation_example_fig
from multabench.leaderboard.text_results                import compute_curation_grid, _load_pool_corpus_data
from multabench.leaderboard.main_paper.text_pool        import make_joint_tar_figure   as _make_text_pool_joint_tar_fig
from multabench.leaderboard.main_paper.text_pool        import make_tfidf_figure       as _make_text_pool_tfidf_fig
from multabench.leaderboard.main_paper.text_pool        import make_struct_unstruct_figure as _make_text_pool_struct_fig
from multabench.leaderboard.main_paper.leaderboard      import make_figure      as _make_leaderboard_fig
from multabench.leaderboard.main_paper.leaderboard      import load_for_overview
from multabench.leaderboard.main_paper.encoder_scale    import make_figure      as _make_encoder_scale_fig
from multabench.leaderboard.main_paper.pca              import make_figure      as _make_pca_fig

# ---------------------------------------------------------------------------
# Constants (for appendix tables only)
# ---------------------------------------------------------------------------

_RESULTS_ROOT = join(dirname(__file__), "results")

_CORE_MODELS = {
    "LightGBM 💡", "CatBoost 😸", "TabM Ⓜ️",
    "TabPFN-v2 🤯", "TabPFN-v2p5 🇩🇪",
}
_E2E_IMAGE_MODELS = {"AutoGluon-MM 🧴"}
_E2E_TEXT_MODELS  = {"TabSTAR ⭐", "ConTextTab 🏢", "AutoGluon-MM 🧴"}

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
    "TabSTAR ⭐": "TabSTAR", "ConTextTab 🏢": "ConTextTab",
    "AutoGluon-MM 🧴": "AG-MM",
    "TabDPT 6️⃣": "TabDPT", "TabICLv2 🗼": "TabICLv2",
    "RandomForest 🌳": "RF", "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost",
}

_DATASET_LABELS = {
    # image
    "BIN_IMAGE_CELEB_ATTRACTIVENESS":  "CelebA Attractiveness",
    "BIN_IMAGE_HATEFUL_MEME":          "Hateful Meme",
    "BIN_IMAGE_MAMMOGRAPHY_CMMD":      "Mammography CMMD",
    "MUL_IMAGE_CBIS_DDSM":             "CBIS-DDSM",
    "MUL_IMAGE_CHEXPERT":              "CheXpert",
    "MUL_IMAGE_CSGO_SKIN_PRICE":       "CS:GO Skins",
    "MUL_IMAGE_FLOWER_BOUQUETS":       "Flower Bouquets",
    "MUL_IMAGE_GLAUCOMA_SMDG":         "Glaucoma SMDG",
    "MUL_IMAGE_HUBMAP_HPA":            "HubMAP HPA",
    "MUL_IMAGE_JUSTIN_INSTAGRAM":      "Justin Instagram",
    "MUL_IMAGE_PETFINDER":             "PetFinder",
    "MUL_IMAGE_ZOOSCAN_ZOOPLANKTON":   "Zooscan Plankton",
    "REG_IMAGE_AMAZON_BEST_SELLER":    "Amazon Bestseller",
    "REG_IMAGE_AMAZON_PACKAGES":       "Amazon Packages",
    "REG_IMAGE_HNM_FASHION":           "H\\&M Fashion",
    "REG_IMAGE_KHAADI_CLOTHES":        "Khaadi Clothes",
    "REG_IMAGE_LETTERBOXD_MOVIES":     "Letterboxd Movies",
    "REG_IMAGE_MANGO_MASS":            "Mango Mass",
    "REG_IMAGE_MKPHOTO_BOTS":          "MkPhoto Bots",
    "REG_IMAGE_PAINTING_PRICE":        "Painting Price",
    # text
    "BIN_TEXT_FAKE_JOB_POSTING":       "Fake Job Postings",
    "BIN_TEXT_JIGSAW_TOXICITY":        "Jigsaw Toxicity",
    "BIN_TEXT_KICKSTARTER_FUNDING":    "Kickstarter",
    "MUL_TEXT_DATA_SCIENTIST_SALARY":  "Data Scientist Salary",
    "MUL_TEXT_MICHELIN_RESTAURANTS":   "Michelin Guide",
    "MUL_TEXT_PRODUCT_SENTIMENT":      "Product Sentiment",
    "MUL_TEXT_SPOTIFY_GENRES":         "Spotify Genres",
    "MUL_TEXT_US_ACCIDENTS":           "US Accidents",
    "MUL_TEXT_WINE_REVIEW":            "Wine Review",
    "MUL_TEXT_WOMEN_CLOTHING_REVIEW":  "Women's Clothing",
    "REG_TEXT_BABIES_PRICES":          "Baby Products",
    "REG_TEXT_BOOK_PRICE":             "Book Price",
    "REG_TEXT_BOOK_READABILITY":       "Book Readability",
    "REG_TEXT_MERCARI_MARKETPLACE":    "Mercari",
    "REG_TEXT_MONTGOMERY_SALARIES":    "Montgomery Salaries",
    "REG_TEXT_ROTTEN_TOMATOES":        "Rotten Tomatoes",
    "REG_TEXT_SCIMAGOJR_IMPACT":       "SciMagojr Impact",
    "REG_TEXT_VANCOUVER_SALARIES":     "Vancouver Salaries",
    "REG_TEXT_VIDEO_GAMES_SALES":      "Video Games Sales",
    "REG_TEXT_ZOMATO_RESTAURANTS":     "Zomato Restaurants",
}

_COSTS_ROOT = join(_RESULTS_ROOT, "costs")
_COSTS_FILES = {
    ("IMAGE", "small"): "image_all.csv",
    ("IMAGE", "large"): "image_large.csv",
    ("TEXT",  "small"): "text_small.csv",
    ("TEXT",  "large"): "text_large.csv",
}
_COSTS_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_PETFINDER_IMG_CSV  = join(_RESULTS_ROOT, "images/MUL_IMAGE_PETFINDER.csv")
_PETFINDER_TRI_CSV  = join(_RESULTS_ROOT, "tabular_image_text/MUL_IMAGE_PETFINDER.csv")
_PETFINDER_COL_ORDER  = ["img", "txt", "non_txt", "non", "all", "ft", "ft-txt", "ft-img-ft-txt"]
_PETFINDER_COL_LABELS = ["I",   "T",   "S+I",    "S+T", "S+I+T", "S+I_TAR+T", "S+I+T_TAR", "S+I_TAR+T_TAR"]
_PETFINDER_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_AMAZON_IMG_CSV    = join(_RESULTS_ROOT, "images/REG_IMAGE_AMAZON_PACKAGES.csv")
_AMAZON_TRI_CSV    = join(_RESULTS_ROOT, "tabular_image_text/REG_IMAGE_AMAZON_PACKAGES.csv")
_AMAZON_COL_ORDER  = ["img", "txt", "non_txt", "non", "all", "ft", "ft-txt", "ft-img-ft-txt"]
_AMAZON_COL_LABELS = ["I",   "T",   "S+I",    "S+T", "S+I+T", "S+I_TAR+T", "S+I+T_TAR", "S+I_TAR+T_TAR"]
_AMAZON_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]


# ---------------------------------------------------------------------------
# Appendix data loading
# ---------------------------------------------------------------------------

def _load_dir(subdir: str) -> pd.DataFrame:
    path = join(_RESULTS_ROOT, subdir)
    frames = []
    for f in listdir(path):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(join(path, f))
        if "dataset" not in df.columns:
            df["dataset"] = f.replace(".csv", "")
        df["model"]      = df["model"].str.strip()
        df["test_score"] = df["test_score"].clip(lower=-0.1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def _load_core_results(modality: str) -> pd.DataFrame:
    subdir = "images" if modality == "IMAGE" else "text"
    df = _load_dir(subdir)
    return df[df["model"].isin(_CORE_MODELS)]


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _fig_to_bytes(fig: plt.Figure, fmt: str = "pdf") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Main-paper tables
# ---------------------------------------------------------------------------

def _make_conditions_table() -> pd.DataFrame:
    return pd.DataFrame({
        "Condition":     ["Unimodal Structured", "Unimodal Unstructured", "Joint Frozen", "Joint TAR"],
        "Structured":    ["✓", "✗", "✓", "✓"],
        "Unstructured":  ["✗", "✓", "✓", "✓"],
        "Target-Aware":  ["--", "✗", "✗", "✓"],
    }).set_index("Condition")


def _to_latex_conditions(tbl: pd.DataFrame) -> str:
    rows = []
    for cond, row in tbl.iterrows():
        s, u, t = row["Structured"], row["Unstructured"], row["Target-Aware"]
        for sym, tex in [("✓", "\\checkmark"), ("✗", "\\times"), ("--", "--")]:
            s, u, t = s.replace(sym, tex), u.replace(sym, tex), t.replace(sym, tex)
        rows.append(f"{cond} & {s} & {u} & {t} \\\\")
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Experimental Conditions. Breakdown by feature composition and representation strategy.}\n"
        "\\label{tab:conditions}\n"
        "\\begin{tabular}{lccc}\n\\hline\n"
        "\\textbf{Condition} & \\textbf{Structured} & \\textbf{Unstructured} & \\textbf{Target-Aware (TAR)} \\\\ \\hline\n"
        + "\n".join(rows) +
        "\n\\hline\n\\end{tabular}\n\\end{table}"
    )


def _make_costs_table(modality: str) -> pd.DataFrame:
    frames = []
    for size in ("small", "large"):
        fname = _COSTS_FILES[(modality, size)]
        df = pd.read_csv(join(_COSTS_ROOT, fname))
        df["encoder_size"] = size
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].str.strip()
    df = df[df["model"].isin(_CORE_MODELS)].copy()
    df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")
    df["train_peak_gpu_gb"] = pd.to_numeric(df["train_peak_gpu_gb"], errors="coerce")
    df["label"] = df["model"].map(_MODEL_LABELS)
    agg = (df.groupby(["label", "encoder_size", "multimodal_state"])
             .agg(runtime=("Runtime", "median"), gpu=("train_peak_gpu_gb", "median"))
             .round(1))
    # pivot to wide: columns = (size, state, metric)
    tbl = agg.unstack(["encoder_size", "multimodal_state"])
    tbl.columns = ["_".join(c) for c in tbl.columns]
    col_order = [
        "runtime_small_all", "runtime_small_ft",
        "gpu_small_all",     "gpu_small_ft",
        "runtime_large_all", "runtime_large_ft",
        "gpu_large_all",     "gpu_large_ft",
    ]
    tbl = tbl[[c for c in col_order if c in tbl.columns]]
    return tbl.reindex(_COSTS_MODEL_ORDER).reset_index()


def _make_costs_figure() -> plt.Figure:
    from matplotlib.patches import Patch
    _C_FROZEN = "#A8D4F0"
    _C_TAR    = "#E8722A"

    conditions = [
        ("IMAGE", "small", "Image\nSmall"),
        ("IMAGE", "large", "Image\nLarge"),
        ("TEXT",  "small", "Text\nSmall"),
        ("TEXT",  "large", "Text\nLarge"),
    ]

    # aggregate median across models for each condition × state
    frames = []
    for (mod, size), fname in _COSTS_FILES.items():
        df = pd.read_csv(join(_COSTS_ROOT, fname))
        df["modality"] = mod
        df["encoder_size"] = size
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].str.strip()
    df = df[df["model"].isin(_CORE_MODELS)].copy()
    df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")
    df["train_peak_gpu_gb"] = pd.to_numeric(df["train_peak_gpu_gb"], errors="coerce")
    agg = (df.groupby(["modality", "encoder_size", "multimodal_state"])
             .agg(runtime=("Runtime", "median"), gpu=("train_peak_gpu_gb", "median"))
             .reset_index())

    x = np.arange(len(conditions))
    w = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for ax, metric, ylabel, log_scale in [
        (axes[0], "runtime", "Runtime (s)", True),
        (axes[1], "gpu",     "Peak GPU memory (GB)", False),
    ]:
        for i, (mod, size, _) in enumerate(conditions):
            for j, (state, color, label) in enumerate([("all", _C_FROZEN, "Frozen"), ("ft", _C_TAR, "TAR")]):
                row = agg[(agg["modality"] == mod) & (agg["encoder_size"] == size) & (agg["multimodal_state"] == state)]
                val = row[metric].values[0] if len(row) else 0
                ax.bar(i + (j - 0.5) * w, val, w, color=color, edgecolor="white", linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in conditions], fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        if log_scale:
            ax.set_yscale("log")
        # faint vertical divider between image and text groups
        ax.axvline(1.5, color="0.75", linewidth=0.8, linestyle="--")

    axes[0].legend(
        handles=[Patch(color=_C_FROZEN, label="Frozen"), Patch(color=_C_TAR, label="TAR")],
        frameon=False, fontsize=13, loc="upper left",
    )
    fig.tight_layout()
    return fig


def _to_latex_costs(tbl_img: pd.DataFrame, tbl_txt: pd.DataFrame) -> str:
    def _rows(tbl):
        lines = []
        for _, r in tbl.iterrows():
            def _rt(col):
                v = r.get(col)
                return "--" if pd.isna(v) else f"{int(v):,}"
            def _gb(col):
                v = r.get(col)
                return "--" if pd.isna(v) else f"{v:.1f}"
            lines.append(
                f"{r['label']} & "
                f"{_rt('runtime_small_all')} & {_rt('runtime_small_ft')} & "
                f"{_gb('gpu_small_all')} & {_gb('gpu_small_ft')} & "
                f"{_rt('runtime_large_all')} & {_rt('runtime_large_ft')} & "
                f"{_gb('gpu_large_all')} & {_gb('gpu_large_ft')} \\\\"
            )
        return "\n".join(lines)

    header = (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{Computation costs per (dataset, fold) run on a single GPU. "
        "Runtime in seconds (median over all datasets and folds). "
        "Peak GPU memory in GB (median). "
        "Frozen = structured + frozen embeddings; "
        "TAR = structured + target-aware fine-tuned embeddings. "
        "Image encoder: DINO-v3-small / DINO-v3-large. "
        "Text encoder: E5-small-v2 / E5-large-v2.}\n"
        "\\label{tab:compute_costs}\n"
        "\\setlength{\\tabcolsep}{4pt}\\small\n"
        "\\begin{tabular}{l rr rr rr rr}\n"
        "\\toprule\n"
        "& \\multicolumn{4}{c}{\\textit{Small Encoder}} "
        "& \\multicolumn{4}{c}{\\textit{Large Encoder}} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "& \\multicolumn{2}{c}{Runtime (s)} & \\multicolumn{2}{c}{Peak GPU (GB)} "
        "& \\multicolumn{2}{c}{Runtime (s)} & \\multicolumn{2}{c}{Peak GPU (GB)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        "\\textbf{Model} & Frozen & TAR & Frozen & TAR & Frozen & TAR & Frozen & TAR \\\\\n"
        "\\midrule\n"
        "\\multicolumn{9}{l}{\\textit{Image-Tabular (DINO encoder)}} \\\\\n"
        "\\midrule\n"
    )
    middle = (
        "\n\\midrule\n"
        "\\multicolumn{9}{l}{\\textit{Text-Tabular (E5 encoder)}} \\\\\n"
        "\\midrule\n"
    )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
    return header + _rows(tbl_img) + middle + _rows(tbl_txt) + footer


def _make_petfinder_table() -> pd.DataFrame:
    img_df = pd.read_csv(_PETFINDER_IMG_CSV)
    tri_df = pd.read_csv(_PETFINDER_TRI_CSV)
    for df in [img_df, tri_df]:
        df.columns = [c.strip() for c in df.columns]
        df["model"] = df["model"].str.strip()
    df = pd.concat([img_df[["model", "multimodal_state", "test_score", "fold"]],
                    tri_df[["model", "multimodal_state", "test_score", "fold"]]],
                   ignore_index=True)
    df["model_label"] = df["model"].map(_MODEL_LABELS)
    pivot = (df.groupby(["model_label", "multimodal_state"])["test_score"]
               .mean()
               .unstack("multimodal_state")
               .multiply(100)
               .round(1))
    tbl = pivot.reindex(index=_PETFINDER_MODEL_ORDER, columns=_PETFINDER_COL_ORDER)
    tbl.columns = _PETFINDER_COL_LABELS
    tbl.index.name = "Model"
    return tbl


def _to_latex_petfinder(tbl: pd.DataFrame) -> str:
    rows = []
    for model, row in tbl.iterrows():
        best = row.max()
        vals = " & ".join(f"\\textbf{{{v:.1f}}}" if v == best else f"{v:.1f}" for v in row)
        rows.append(f"{model} & {vals} \\\\")
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{The PetFinder Analysis. S=Structured, I=Image, T=Text. "
        "AUC (\\%) per model-condition pair. "
        "The best condition performs Joint Modeling and Target-Aware Representations for both modalities.}\n"
        "\\label{tab:petfinder}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\small\n"
        "\\begin{tabular}{l cc ccc ccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{\\textit{Single modality}} "
        "& \\multicolumn{3}{c}{\\textit{Frozen combinations}} "
        "& \\multicolumn{3}{c}{\\textit{Target-Aware Representations (TAR)}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
        "\\textbf{Model} & I & T & S+I & S+T & S+I+T "
        "& S+I\\textsubscript{TAR}+T & S+I+T\\textsubscript{TAR} "
        "& \\textbf{S+I\\textsubscript{TAR}+T\\textsubscript{TAR}} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


def _make_amazon_packages_table() -> pd.DataFrame:
    img_df = pd.read_csv(_AMAZON_IMG_CSV)
    tri_df = pd.read_csv(_AMAZON_TRI_CSV)
    for df in [img_df, tri_df]:
        df.columns = [c.strip() for c in df.columns]
        df["model"] = df["model"].str.strip()
    df = pd.concat([img_df[["model", "multimodal_state", "test_score", "fold"]],
                    tri_df[["model", "multimodal_state", "test_score", "fold"]]],
                   ignore_index=True)
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"])
    pivot = (df.groupby(["model_label", "multimodal_state"])["test_score"]
               .mean()
               .unstack("multimodal_state")
               .multiply(100)
               .round(1))
    tbl = pivot.reindex(index=_AMAZON_MODEL_ORDER, columns=_AMAZON_COL_ORDER)
    tbl.columns = _AMAZON_COL_LABELS
    tbl.index.name = "Model"
    return tbl


def _to_latex_amazon_packages(tbl: pd.DataFrame) -> str:
    rows = []
    for model, row in tbl.iterrows():
        best = row.max()
        vals = " & ".join(f"\\textbf{{{v:.1f}}}" if v == best else f"{v:.1f}" for v in row)
        rows.append(f"{model} & {vals} \\\\")
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Amazon Packages trimodal analysis. S=Structured, I=Image, T=Text. "
        "Mean $R^2$ (\\%) per model and condition. "
        "The best condition uses Target-Aware Representations for both modalities.}\n"
        "\\label{tab:amazon_packages}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\small\n"
        "\\begin{tabular}{l cc ccc ccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{\\textit{Single modality}} "
        "& \\multicolumn{3}{c}{\\textit{Frozen combinations}} "
        "& \\multicolumn{3}{c}{\\textit{Target-Aware Representations (TAR)}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
        "\\textbf{Model} & I & T & S+I & S+T & S+I+T "
        "& S+I\\textsubscript{TAR}+T & S+I+T\\textsubscript{TAR} "
        "& \\textbf{S+I\\textsubscript{TAR}+T\\textsubscript{TAR}} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------

def _make_results_table(modality: str) -> pd.DataFrame:
    subdir = "images" if modality == "IMAGE" else "text"
    base_df = _load_dir(subdir)
    mb_df   = _load_dir("more_baselines")
    ref_datasets = set(base_df["dataset"].unique())
    mb_df = mb_df[mb_df["dataset"].isin(ref_datasets)]
    df = pd.concat([base_df, mb_df], ignore_index=True)

    # keep only models that have both frozen (all) and TAR (ft) on this modality
    has_both = (df.groupby("model")["multimodal_state"]
                  .apply(lambda s: {"all", "ft"}.issubset(set(s)))
                  .loc[lambda x: x].index)
    df = df[df["model"].isin(has_both)]

    n_models = df["model"].nunique()
    avg = (df.groupby(["dataset", "multimodal_state"])["test_score"]
             .mean().unstack("multimodal_state"))
    tbl = pd.DataFrame({
        "Dataset": [_DATASET_LABELS.get(i, i) for i in avg.index],
        "Task":    [i.split("_")[0] for i in avg.index],
        "Frozen":  avg.get("all"),
        "FT":      avg.get("ft"),
    })
    tbl["Gain"] = (tbl["FT"] - tbl["Frozen"]).round(3)
    tbl[["Frozen", "FT"]] = tbl[["Frozen", "FT"]].round(3)
    tbl.attrs["n_models"] = n_models
    return tbl.sort_values("Gain", ascending=False).reset_index(drop=True)


def _to_latex(tbl: pd.DataFrame, label: str, caption: str) -> str:
    header = (
        "\\begin{table*}[h]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Dataset & Task & Frozen & Contextualized & Gain \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for _, r in tbl.iterrows():
        gain_str = f"$+${r['Gain']:.3f}" if r['Gain'] >= 0 else f"$-${abs(r['Gain']):.3f}"
        rows.append(
            f"{r['Dataset']} & {r['Task']} & {r['Frozen']:.3f}"
            f" & {r['FT']:.3f} & {gain_str} \\\\"
        )
    mean_row = (
        f"\\midrule\n"
        f"\\textit{{Mean}} & & {tbl['Frozen'].mean():.3f}"
        f" & {tbl['FT'].mean():.3f} & $+${tbl['Gain'].mean():.3f} \\\\"
    )
    return header + "\n".join(rows) + "\n" + mean_row + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"


def _make_win_rate_table() -> pd.DataFrame:
    records = []
    for modality in ["IMAGE", "TEXT"]:
        e2e_raw    = _E2E_IMAGE_MODELS if modality == "IMAGE" else _E2E_TEXT_MODELS
        e2e_labels = {_MODEL_LABELS.get(m, m) for m in e2e_raw}

        df = load_for_overview(modality)
        df["label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"])
        df = df[~df["label"].isin(e2e_labels)]

        pivot = (df.pivot_table(index=["label", "dataset", "fold"],
                                columns="multimodal_state", values="test_score")
                   .dropna(subset=["all", "ft"]))
        pivot["ft_wins"] = (pivot["ft"] > pivot["all"]).astype(float)
        stats = pivot.groupby("label")["ft_wins"].agg(["mean", "count"])
        for model, row in stats.iterrows():
            p, n = row["mean"], row["count"]
            ci = 1.96 * np.sqrt(p * (1 - p) / n) * 100
            records.append({"Model": model, "modality": modality,
                            "wr": round(p * 100, 1), "ci": round(ci, 1)})

    df_r = pd.DataFrame(records)
    wr = df_r.pivot(index="Model", columns="modality", values="wr").rename(
        columns={"IMAGE": "Image", "TEXT": "Text"})
    ci = df_r.pivot(index="Model", columns="modality", values="ci").rename(
        columns={"IMAGE": "Image CI", "TEXT": "Text CI"})
    tbl = wr.join(ci)
    tbl["Combined"]    = tbl[["Image", "Text"]].mean(axis=1).round(1)
    tbl["Combined CI"] = (tbl[["Image CI", "Text CI"]].pow(2).mean(axis=1) ** 0.5).round(1)
    col_order = ["Image", "Image CI", "Text", "Text CI", "Combined", "Combined CI"]
    return tbl[[c for c in col_order if c in tbl.columns]].sort_values(
        "Combined", ascending=False).reset_index()


@st.cache_data
def _make_datasets_table() -> pd.DataFrame:
    df = pd.read_csv(_SUMMARY_CSV)
    df["Classes"] = df["Classes"].apply(lambda x: str(int(x)) if pd.notna(x) else "--")
    return df


def _get_datasets_table_latex() -> str:
    tex_path = expanduser("~/tabstar/paper-multabench/appendix.tex")
    with open(tex_path) as f:
        content = f.read()
    start = content.find("\\begin{table*}")
    end   = content.find("\\end{table*}") + len("\\end{table*}")
    return content[start:end]


def _to_latex_win_rate(tbl: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Per-model FT win rate on MulTaBench. "
        "Win rate = fraction of (dataset, fold) pairs where FT $>$ Frozen, with 95\\% CI. "
        "End-to-end models excluded. Combined = mean of Image and Text.}\n"
        "\\label{tab:win_rate}\n\\small\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Model & Image (\\%) & Text (\\%) & Combined (\\%) \\\\\n"
        "\\midrule\n"
    )

    def _fmt(wr, ci):
        return "" if pd.isna(wr) else f"${wr:.1f} \\pm {ci:.1f}$"

    rows = [
        f"{r['Model']} & {_fmt(r.get('Image'), r.get('Image CI', 0))} "
        f"& {_fmt(r.get('Text'), r.get('Text CI', 0))} "
        f"& \\textbf{{{_fmt(r.get('Combined'), r.get('Combined CI', 0))}}} \\\\"
        for _, r in tbl.iterrows()
    ]
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"


def _make_curation_grid_latex(grid: pd.DataFrame) -> str:
    models = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]
    header = (
        "\\begin{longtable}{lcccccc}\n"
        "\\caption{Per-dataset curation grid across all 56 text-tabular candidates. "
        "Each cell indicates whether that model satisfies all three criteria: "
        "Joint Signal and Task-awareness (\\S\\ref{sec:benchmarking}). "
        "Datasets are sorted approved-first, then by descending pass count.}"
        "\\label{tab:text_curation_grid}\\\\\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{LGB} & \\textbf{CTB} & \\textbf{TabM} "
        "& \\textbf{PFNv2} & \\textbf{PFN-2.5} & \\textbf{Pass} \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{LGB} & \\textbf{CTB} & \\textbf{TabM} "
        "& \\textbf{PFNv2} & \\textbf{PFN-2.5} & \\textbf{Pass} \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\\multicolumn{7}{r}{\\textit{Continued on next page}}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
    )
    rows = []
    prev_decision = None
    for ds, row in grid.iterrows():
        decision = row["decision"]
        if prev_decision is not None and decision != prev_decision:
            rows.append("\\midrule")
        prev_decision = decision
        name = ds.replace("_", "\\_")
        def _cell(v):
            import math
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "$-$"
            return "$\\checkmark$" if v else "$\\times$"
        cells = " & ".join(_cell(row[m]) for m in models)
        count = int(row["all_three (/5)"])
        rows.append(f"{name} & {cells} & {count} \\\\")
    return header + "\n".join(rows) + "\n\\end{longtable}"


def _make_text_curation_rates_table() -> pd.DataFrame:
    benchmarks = [
        ("AutoML Multimodal", AUTOML_MULTIMODAL_ACCEPTED, AUTOML_MULTIMODAL_REJECTED),
        ("Grinsztajn et al.", VECTORIZING_ACCEPTED,       VECTORIZING_REJECTED),
        ("CARTE",             CARTE_ACCEPTED,              CARTE_REJECTED),
        ("TextTabBench",      TEXT_TAB_BENCH_ACCEPTED,     TEXT_TAB_BENCH_REJECTED),
    ]
    rows = []
    for name, acc, rej in benchmarks:
        candidates = len(set(acc + rej))
        accepted   = len(set(acc))
        rows.append({"Benchmark": name, "Candidates": candidates,
                     "Accepted": accepted, "Rate": f"{accepted / candidates:.0%}"})
    total_cands = len(set(ACCEPTED_TEXT_DATASETS) | set(REJECTED_TEXT_DATASETS))
    total_acc   = len(set(ACCEPTED_TEXT_DATASETS))
    rows.append({"Benchmark": "Total (unique, deduplicated)", "Candidates": total_cands,
                 "Accepted": total_acc, "Rate": f"{total_acc / total_cands:.0%}"})
    return pd.DataFrame(rows)


def _to_latex_text_curation_rates(tbl: pd.DataFrame) -> str:
    header = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Text-tabular curation acceptance rates by source benchmark. "
        "Counts refer to unique datasets evaluated from each source.}\n"
        "\\label{tab:text_curation_rates}\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "\\textbf{Benchmark} & \\textbf{Candidates} & \\textbf{Accepted} & \\textbf{Rate} \\\\\n"
        "\\midrule\n"
    )
    cite_map = {
        "AutoML Multimodal": "AutoML Multimodal \\cite{shi_benchmarking_2021}",
        "Grinsztajn et al.": "\\citet{grinsztajn_vectorizing_2023}",
        "CARTE":             "CARTE \\cite{kim_carte_2024}",
        "TextTabBench":      "TextTabBench \\cite{mraz_towards_2025}",
        "Total (unique, deduplicated)": "\\textbf{Total (unique, deduplicated)}",
    }
    rows = []
    for _, r in tbl.iterrows():
        name = cite_map.get(r["Benchmark"], r["Benchmark"])
        acc  = r["Accepted"]
        cand = r["Candidates"]
        rate = r["Rate"]
        rate_tex = rate.replace("%", "\\%")
        bold = r["Benchmark"].startswith("Total")
        if bold:
            rows.append(f"{name} & \\textbf{{{cand}}} & \\textbf{{{acc}}} & \\textbf{{{rate_tex}}} \\\\")
        else:
            rows.append(f"{name} & {cand} & {acc} & {rate_tex} \\\\")
    body = "\n".join(rows[:4]) + "\n\\midrule\n" + rows[4]
    return header + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"


# ---------------------------------------------------------------------------
# Streamlit display
# ---------------------------------------------------------------------------

def display_paper_production():
    st.title("📄 Paper Production")
    st.caption("Generate and download paper-quality figures and tables from live data.")

    main_tab, tbl_tab, app_tab = st.tabs(["📄 Main Paper", "📋 Tables", "📎 Appendix"])

    # ── Main Paper ────────────────────────────────────────────────────────────
    with main_tab:
        _MAIN_PAPER = [
            ("Figure 2", "Curation Protocol over Candidate Datasets", "figure",
             lambda: (_make_curation_example_fig(), None),          "curation_example"),
            ("Figure 3", "Target-Aware Representations Gains over Frozen", "figure",
             _make_text_pool_joint_tar_fig,                         "text_pool_joint_tar"),
            ("Table 2",  "The PetFinder Analysis",                  "table",
             _make_petfinder_table,                                 "petfinder"),
            ("Figure 4", "Tabular Learners Performance Analysis",   "figure",
             _make_leaderboard_fig,                                 "leaderboard"),
            ("Figure 5", "Embedding Model Size Analysis",           "figure",
             lambda: _make_encoder_scale_fig("all"),                "encoder_scale"),
            ("Figure 6", "PCA Projection Dimensions",               "figure",
             _make_pca_fig,                                         "pca"),
        ]

        def _show_agg(agg_data):
            if agg_data is None:
                return
            with st.expander("Aggregated data (used to draw the plot)"):
                if isinstance(agg_data, dict):
                    for panel_name, df_agg in agg_data.items():
                        st.caption(panel_name)
                        st.dataframe(df_agg, use_container_width=True)
                else:
                    st.dataframe(agg_data, use_container_width=True)

        st.subheader("Table 1 — Experimental Conditions")
        cond_tbl   = _make_conditions_table()
        cond_latex = _to_latex_conditions(cond_tbl)
        st.dataframe(cond_tbl, use_container_width=True)
        with st.expander("📋 LaTeX source"):
            st.code(cond_latex, language="latex")
        st.divider()

        for ref, title, kind, make_fn, fname_stem in _MAIN_PAPER:
            st.subheader(f"{ref} — {title}")
            if kind == "figure":
                _fig, _agg = make_fn()
                st.pyplot(_fig, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇ Download PDF", data=_fig_to_bytes(_fig, "pdf"),
                                       file_name=f"{fname_stem}.pdf", mime="application/pdf",
                                       key=f"dl_pdf_{fname_stem}")
                with c2:
                    st.download_button("⬇ Download PNG", data=_fig_to_bytes(_fig, "png"),
                                       file_name=f"{fname_stem}.png", mime="image/png",
                                       key=f"dl_png_{fname_stem}")
                _show_agg(_agg)
                plt.close(_fig)
            else:
                _tbl = make_fn()
                st.dataframe(
                    _tbl.style.highlight_max(axis=1, props="font-weight:bold").format("{:.1f}"),
                    use_container_width=True)
                _latex = _to_latex_petfinder(_tbl)
                with st.expander("📋 LaTeX source"):
                    st.code(_latex, language="latex")
            st.divider()

    # ── Tables ────────────────────────────────────────────────────────────────
    with tbl_tab:
        st.subheader("Appendix — Per-Dataset Results")
        _APP_CAPTIONS = {
            "IMAGE": (
                "MulTaBench image-tabular benchmark: per-dataset results averaged over "
                "5 core learners and 5 random seeds, sorted by Gain. "
                "Frozen: structured + frozen DINO-v3 embeddings. "
                "Contextualized: structured + fine-tuned DINO-v3 embeddings. "
                "Gain: Contextualized $-$ Frozen. "
                "AUROC for classification ($\\uparrow$), $R^2$ for regression ($\\uparrow$). "
                "$R^2$ clipped at $-0.1$.",
                "tab:app_results_image",
            ),
            "TEXT": (
                "MulTaBench text-tabular benchmark: per-dataset results averaged over "
                "5 core learners and 5 random seeds, sorted by Gain. "
                "Frozen: structured + frozen E5-Small embeddings. "
                "Contextualized: structured + fine-tuned E5-Small embeddings. "
                "Gain: Contextualized $-$ Frozen. "
                "AUROC for classification ($\\uparrow$), $R^2$ for regression ($\\uparrow$). "
                "$R^2$ clipped at $-0.1$.",
                "tab:app_results_text",
            ),
        }
        for modality, title in [("IMAGE", "Image-Tabular Results"), ("TEXT", "Text-Tabular Results")]:
            st.subheader(title)
            tbl = _make_results_table(modality)
            st.dataframe(tbl.style.background_gradient(subset=["Gain"], cmap="RdYlGn")
                         .format({"Frozen": "{:.3f}", "FT": "{:.3f}", "Gain": "{:+.3f}"}),
                         use_container_width=True)
            caption, label = _APP_CAPTIONS[modality]
            latex = _to_latex(tbl, label, caption)
            st.download_button("⬇ Download CSV", data=tbl.to_csv(index=False),
                               file_name=f"results_{modality.lower()}.csv", mime="text/csv",
                               key=f"dl_csv_{modality}")
            with st.expander("📋 LaTeX source"):
                st.code(latex, language="latex")
            st.divider()

        st.subheader("Win Rate by Model")
        wr_tbl   = _make_win_rate_table()
        wr_latex = _to_latex_win_rate(wr_tbl)
        display_cols = [c for c in ["Model", "Image", "Image CI", "Text", "Text CI",
                                    "Combined", "Combined CI"] if c in wr_tbl.columns]
        fmt = {c: "{:.1f}" for c in display_cols if c != "Model"}
        st.dataframe(wr_tbl[display_cols].style.background_gradient(
            subset=["Image", "Text", "Combined"], cmap="RdYlGn", vmin=0, vmax=100
        ).format(fmt, na_rep=""), use_container_width=True)
        st.download_button("⬇ Download CSV", data=wr_tbl.to_csv(index=False),
                           file_name="win_rate.csv", mime="text/csv", key="dl_csv_wr")
        with st.expander("📋 LaTeX source"):
            st.code(wr_latex, language="latex")

    # ── Appendix ──────────────────────────────────────────────────────────────
    with app_tab:
        st.subheader("Table 3 — MulTaBench Datasets")
        datasets_df = _make_datasets_table()
        st.dataframe(datasets_df, use_container_width=True, hide_index=True)
        datasets_latex = _get_datasets_table_latex()
        with st.expander("📋 LaTeX source"):
            st.code(datasets_latex, language="latex")
        st.divider()

        st.subheader("Table — Text Curation Grid (56 datasets × 5 models)")
        _pool_df = _load_pool_corpus_data()
        from multabench.leaderboard.text_results import single_variant_name
        from multabench.leaderboard.data.keys import MODE
        _pool_df[MODE] = _pool_df.apply(single_variant_name, axis=1)
        _grid_df = compute_curation_grid(_pool_df)
        st.dataframe(_grid_df, use_container_width=True)
        _grid_latex = _make_curation_grid_latex(_grid_df)
        with st.expander("📋 LaTeX source (longtable)"):
            st.code(_grid_latex, language="latex")
        st.divider()

        st.subheader("Table — Text Curation Acceptance Rates")
        curation_df = _make_text_curation_rates_table()
        st.dataframe(curation_df, use_container_width=True, hide_index=True)
        curation_latex = _to_latex_text_curation_rates(curation_df)
        with st.expander("📋 LaTeX source"):
            st.code(curation_latex, language="latex")
        st.divider()

        st.subheader("Table — Computation Costs")
        costs_img = _make_costs_table("IMAGE")
        costs_txt = _make_costs_table("TEXT")
        st.caption("Image (DINO)")
        st.dataframe(costs_img, use_container_width=True, hide_index=True)
        st.caption("Text (E5)")
        st.dataframe(costs_txt, use_container_width=True, hide_index=True)
        costs_latex = _to_latex_costs(costs_img, costs_txt)
        with st.expander("📋 LaTeX source"):
            st.code(costs_latex, language="latex")

        st.subheader("Figure — Computation Costs")
        costs_fig = _make_costs_figure()
        st.pyplot(costs_fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download PDF", data=_fig_to_bytes(costs_fig, "pdf"),
                               file_name="compute_costs.pdf", mime="application/pdf",
                               key="dl_pdf_compute_costs")
        with c2:
            st.download_button("⬇ Download PNG", data=_fig_to_bytes(costs_fig, "png"),
                               file_name="compute_costs.png", mime="image/png",
                               key="dl_png_compute_costs")
        plt.close(costs_fig)
        st.divider()

        st.subheader("Table — Amazon Packages Trimodal Analysis (appendix Tab. amazon_packages)")
        amz_tbl = _make_amazon_packages_table()
        st.dataframe(amz_tbl.style.highlight_max(axis=1, props="font-weight:bold").format("{:.1f}"),
                     use_container_width=True)
        with st.expander("📋 LaTeX source"):
            st.code(_to_latex_amazon_packages(amz_tbl), language="latex")
        st.divider()

        _APP_PLOTS = [
            ("Text Pool: TF-IDF vs E5",              _make_text_pool_tfidf_fig,   "text_pool_tfidf"),
            ("Text Pool: Structured vs Unstructured", _make_text_pool_struct_fig,  "text_pool_struct_unstruct"),
        ]
        for _title, _fn, _stem in _APP_PLOTS:
            st.subheader(_title)
            _fig, _agg = _fn()
            st.pyplot(_fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇ Download PDF", data=_fig_to_bytes(_fig, "pdf"),
                                   file_name=f"{_stem}.pdf", mime="application/pdf",
                                   key=f"dl_pdf_app_{_stem}")
            with c2:
                st.download_button("⬇ Download PNG", data=_fig_to_bytes(_fig, "png"),
                                   file_name=f"{_stem}.png", mime="image/png",
                                   key=f"dl_png_app_{_stem}")
            _show_agg(_agg)
            plt.close(_fig)
            st.divider()

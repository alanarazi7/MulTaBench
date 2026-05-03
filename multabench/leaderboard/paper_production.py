"""
Production exports for the MulTaBench paper.
Generates paper-quality figures (PNG/PDF) and LaTeX tables from live data.
"""
import io
from os import listdir
from os.path import dirname, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from multabench.leaderboard.main_paper.curation_example import make_fig        as _make_curation_example_fig
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

_PETFINDER_IMG_CSV  = join(_RESULTS_ROOT, "images/MUL_IMAGE_PETFINDER.csv")
_PETFINDER_TRI_CSV  = join(_RESULTS_ROOT, "tabular_image_text/MUL_IMAGE_PETFINDER.csv")
_PETFINDER_COL_ORDER  = ["img", "txt", "non_txt", "non", "all", "ft", "ft-txt", "ft-img-ft-txt"]
_PETFINDER_COL_LABELS = ["I",   "T",   "S+I",    "S+T", "S+I+T", "S+I_TAR+T", "S+I+T_TAR", "S+I_TAR+T_TAR"]
_PETFINDER_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]


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


def _make_petfinder_table() -> pd.DataFrame:
    img_df = pd.read_csv(_PETFINDER_IMG_CSV)
    tri_df = pd.read_csv(_PETFINDER_TRI_CSV)
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


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------

def _make_results_table(modality: str) -> pd.DataFrame:
    df = _load_core_results(modality)
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
                st.download_button("⬇ Download LaTeX", data=_latex,
                                   file_name=f"table_{fname_stem}.tex", mime="text/plain",
                                   key=f"dl_tex_{fname_stem}")
                with st.expander("📋 LaTeX source"):
                    st.code(_latex, language="latex")
            st.divider()

    # ── Tables ────────────────────────────────────────────────────────────────
    with tbl_tab:
        st.subheader("Table 1 — Experimental Conditions")
        cond_tbl   = _make_conditions_table()
        cond_latex = _to_latex_conditions(cond_tbl)
        st.dataframe(cond_tbl, use_container_width=True)
        st.download_button("⬇ Download LaTeX", data=cond_latex,
                           file_name="table_conditions.tex", mime="text/plain",
                           key="dl_tex_conditions")
        with st.expander("📋 LaTeX source"):
            st.code(cond_latex, language="latex")
        st.divider()

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
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇ Download CSV", data=tbl.to_csv(index=False),
                                   file_name=f"results_{modality.lower()}.csv", mime="text/csv",
                                   key=f"dl_csv_{modality}")
            with c2:
                st.download_button("⬇ Download LaTeX", data=latex,
                                   file_name=f"table_{modality.lower()}.tex", mime="text/plain",
                                   key=f"dl_tex_{modality}")
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
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download CSV", data=wr_tbl.to_csv(index=False),
                               file_name="win_rate.csv", mime="text/csv", key="dl_csv_wr")
        with c2:
            st.download_button("⬇ Download LaTeX", data=wr_latex,
                               file_name="table_win_rate.tex", mime="text/plain", key="dl_tex_wr")
        with st.expander("📋 LaTeX source"):
            st.code(wr_latex, language="latex")

    # ── Appendix ──────────────────────────────────────────────────────────────
    with app_tab:
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

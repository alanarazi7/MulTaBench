"""Generate LaTeX tables for the app:trimodal_datasets appendix section.

Reads results from:
  multimodal/leaderboard/results/images/<DS>.csv       -- img, ft, non, all states
  multimodal/leaderboard/results/tabular_image_text/<DS>.csv -- non_txt, txt, ft-txt, ft-img-ft-txt

Prints LaTeX tabular rows for each dataset.
"""
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "multimodal/leaderboard/results")

_MODEL_MAP = {
    "LightGBM 💡": "LightGBM",
    "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}
_MODELS_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_STATE_MAP_MAIN   = {"img": "I", "non": "S+T", "all": "S+I+T", "ft": "S+I_TAR+T"}
_STATE_MAP_TRIPLE = {"non_txt": "S+I", "txt": "T", "ft-txt": "S+I+T_TAR", "ft-img-ft-txt": "S+I_TAR+T_TAR"}
_COLS_ORDER = ["I", "T", "S+I", "S+T", "S+I+T", "S+I_TAR+T", "S+I+T_TAR", "S+I_TAR+T_TAR"]


def _load(ds: str) -> pd.DataFrame:
    df_img = pd.read_csv(os.path.join(_RESULTS, "images", f"{ds}.csv"))
    df_tri = pd.read_csv(os.path.join(_RESULTS, "tabular_image_text", f"{ds}.csv"))
    for df in [df_img, df_tri]:
        df["m"] = df["model"].map(_MODEL_MAP)
    df_img["cond"] = df_img["multimodal_state"].map(_STATE_MAP_MAIN)
    df_tri["cond"] = df_tri["multimodal_state"].map(_STATE_MAP_TRIPLE)
    combined = pd.concat([df_img[["m", "cond", "test_score"]], df_tri[["m", "cond", "test_score"]]])
    combined = combined[combined["cond"].notna()]
    pivot = combined.groupby(["m", "cond"])["test_score"].mean().unstack()
    pivot = (pivot * 100).reindex(columns=_COLS_ORDER).reindex(_MODELS_ORDER)
    return pivot


def _latex_rows(pivot: pd.DataFrame, bold_col: str = "S+I_TAR+T_TAR") -> str:
    lines = []
    for model, row in pivot.iterrows():
        vals = []
        for col in _COLS_ORDER:
            v = row[col]
            s = f"{v:.1f}" if pd.notna(v) else "--"
            if col == bold_col:
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        lines.append(f"{model:<12} & " + " & ".join(vals) + " \\\\")
    return "\n".join(lines)


def _check_criteria(ds: str) -> None:
    df_img = pd.read_csv(os.path.join(_RESULTS, "images", f"{ds}.csv"))
    df_tri = pd.read_csv(os.path.join(_RESULTS, "tabular_image_text", f"{ds}.csv"))
    for df in [df_img, df_tri]:
        df["m"] = df["model"].map(_MODEL_MAP)
    p_img = df_img.groupby(["m", "multimodal_state"])["test_score"].mean().unstack()
    p_tri = df_tri.groupby(["m", "multimodal_state"])["test_score"].mean().unstack()
    models = _MODELS_ORDER
    all_frozen = p_img.reindex(models)["all"]
    txt_only   = p_tri.reindex(models)["txt"]
    no_txt     = p_tri.reindex(models)["non_txt"]
    ft_txt     = p_tri.reindex(models)["ft-txt"]
    c1 = ft_txt > all_frozen
    c2 = all_frozen > no_txt
    c3 = all_frozen > txt_only
    n = (c1 & c2 & c3).sum()
    print(f"  {ds}: all_three={n}/5 -> {'PASS' if n >= 3 else 'FAIL'}")


if __name__ == "__main__":
    # Check which datasets pass
    datasets = [
        "MUL_IMAGE_PETFINDER",
        "REG_IMAGE_HNM_FASHION",
        "REG_IMAGE_AMAZON_PACKAGES",
        "REG_IMAGE_LETTERBOXD_MOVIES",
        "MUL_IMAGE_FLOWER_BOUQUETS",
        "REG_IMAGE_KHAADI_CLOTHES",
        "REG_IMAGE_PAINTING_PRICE",
        "MUL_IMAGE_CSGO_SKIN_PRICE",
    ]
    print("=== Trimodal curation criterion (all three simultaneously, >=3/5) ===")
    for ds in datasets:
        _check_criteria(ds)

    print("\n=== Amazon Packages table (R^2 x100) ===")
    pivot = _load("REG_IMAGE_AMAZON_PACKAGES")
    print(pivot.round(1).to_string())
    print()
    print(_latex_rows(pivot))

    print("\n=== PetFinder table (AUC x100) ===")
    pivot = _load("MUL_IMAGE_PETFINDER")
    print(pivot.round(1).to_string())
    print()
    print(_latex_rows(pivot))

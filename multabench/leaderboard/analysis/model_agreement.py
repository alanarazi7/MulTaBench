"""
Model committee agreement analysis @ δ=0.001

Computes pairwise agreement on pass/fail decisions across the 56-dataset text-tabular pool.
Agreement is measured as the percentage of datasets where two models agree on whether
both conditions hold:
  - Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only)) > δ
  - Delta_Awareness = mean(ft) - mean(all) > δ

Run standalone: `python -m multabench.leaderboard.analysis.model_agreement`
"""
from pathlib import Path

import pandas as pd
import numpy as np

from multabench.leaderboard.analysis.committee_pool import build_long_csv, build_pass_matrix


def compute_pairwise_agreement(delta: float = 0.001) -> pd.DataFrame:
    """Compute 10x10 symmetric agreement matrix.

    Args:
        delta: Pass threshold for both Delta_Joint and Delta_Awareness.

    Returns:
        DataFrame with model names as index/columns, agreement % as values.
        Diagonal is NaN.
    """
    df = build_long_csv()
    matrix = build_pass_matrix(df, delta=delta)
    models = list(matrix.columns)

    agreement_vals = np.zeros((len(models), len(models)))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                agreement_vals[i, j] = np.nan
            else:
                pair = matrix[[m1, m2]].dropna()
                if len(pair) == 0:
                    agreement_vals[i, j] = np.nan
                else:
                    same = (pair[m1] == pair[m2]).sum()
                    agreement_vals[i, j] = same / len(pair) * 100

    return pd.DataFrame(agreement_vals, index=models, columns=models)


def plot_agreement_matrix(agreement_df: pd.DataFrame, output_path: str,
                          diagonal_only: bool = False):
    """Visualize agreement matrix as heatmap with annotations.

    Args:
        agreement_df: DataFrame from compute_pairwise_agreement().
        output_path: Path to save PNG.
        diagonal_only: If True, show lower triangle only (excludes upper triangle and diagonal).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    df = agreement_df.values
    models = list(agreement_df.index)
    n = len(models)

    if diagonal_only:
        # Lower triangle only
        df_display = df.copy()
        df_display = np.triu(df_display, k=1)  # Zero out upper triangle + diagonal
        df_display[np.where(df_display == 0)] = np.nan
    else:
        df_display = df

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(df_display, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')

    # Annotations
    for i in range(n):
        for j in range(n):
            val = df_display[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.0f}', ha="center", va="center",
                              color="black" if 60 < val < 80 else "white",
                              fontweight='bold', fontsize=11)
            elif i == j:
                # Diagonal: black square
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True,
                                          facecolor='black', edgecolor='white', linewidth=0.8))

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    title = "Model Agreement Matrix (Lower Triangle) @ δ=0.001" if diagonal_only \
            else "Model Agreement Matrix @ δ=0.001"
    ax.set_title(f'{title}\n(% agreement on pass/fail)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.colorbar(im, ax=ax, label='Agreement (%)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    agreement_df = compute_pairwise_agreement(delta=0.001)

    # Markdown table output
    print("\n" + "="*80)
    print("PAIRWISE AGREEMENT (top 10 pairs)")
    print("="*80)
    pairs = []
    for i, m1 in enumerate(agreement_df.index):
        for j, m2 in enumerate(agreement_df.columns):
            if i < j:
                val = agreement_df.iloc[i, j]
                if not np.isnan(val):
                    pairs.append((m1, m2, val))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for rank, (m1, m2, agreement) in enumerate(pairs[:10], 1):
        print(f"{rank:2d}. {m1:15s} ↔ {m2:15s}  {agreement:6.1f}%")

    # Plot outputs
    results_dir = Path(__file__).parent.parent.parent / "results" / "analysis_curation_sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_agreement_matrix(agreement_df, str(results_dir / "model_agreement_full.png"),
                         diagonal_only=False)
    plot_agreement_matrix(agreement_df, str(results_dir / "model_agreement_diagonal.png"),
                         diagonal_only=True)

import numpy as np
from sklearn.decomposition import PCA


def log_pca_variance(pca: PCA, col_name: str) -> None:
    cum_var = max(np.cumsum(pca.explained_variance_ratio_))
    print(f"🔍 [PCA variance] {col_name}: {cum_var:.3f}")

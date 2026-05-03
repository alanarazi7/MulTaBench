from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from tabstar.constants import SEED
from multabench.baselines.abstract_model import TabularModel


class LightGBM(TabularModel):

    MODEL_NAME = "LightGBM 💡"
    SHORT_NAME = "light"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self) -> LGBMRegressor | LGBMClassifier:
        model_cls = LGBMClassifier if self.is_cls else LGBMRegressor
        params = {"verbose": -1, "random_state": SEED}
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        # Use column names so we only mark truly categorical columns; indices can point to PCA cols after transforms.
        cat_cols = [c for c in x_train.columns if c in self.categorical_features]
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=cat_cols)
        # TOOD: this sohuld be optional
        self.feature_importances_ = self.get_feature_importance()
        print(f"🔥 Feature importances: {self.feature_importances_}")

    def get_feature_importance(self, importance_type: str = "gain") -> Series:
        """
        Get feature importance from the fitted LightGBM model.
        
        Args:
            importance_type: Type of importance to return. Options:
                - "gain": Total gain of splits which use the feature (default)
                - "split": Number of times the feature is used in a model
                
        Returns:
            Series with feature names as index and normalized importance values that sum to 1.
            PCA features are grouped by their prefix (e.g., all "captions_txt_pca_*" become "captions_txt").
        """
        if not hasattr(self.model_, 'booster_'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get feature names from the model (LightGBM stores them automatically from DataFrame columns)
        if hasattr(self.model_, 'feature_name_') and self.model_.feature_name_:
            feature_names = self.model_.feature_name_
        else:
            # Fallback: if feature names aren't available, use indices
            n_features = len(self.model_.feature_importances_)
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Use booster to get importance with specified type
        importances = self.model_.booster_.feature_importance(importance_type=importance_type)
        raw_importance = Series(importances, index=feature_names, name=f"feature_importance_{importance_type}")
        
        # Group PCA features by prefix (everything before "_pca_")
        grouped_importance = {}
        for feature_name, importance_value in raw_importance.items():
            if "_pca_" in feature_name:
                # Extract prefix (everything before "_pca_")
                prefix = feature_name.split("_pca_")[0]
                grouped_importance[prefix] = grouped_importance.get(prefix, 0) + importance_value
            else:
                # Keep non-PCA features as-is
                grouped_importance[feature_name] = grouped_importance.get(feature_name, 0) + importance_value
        
        # Convert to Series and normalize so values sum to 1
        grouped_series = Series(grouped_importance)
        total = grouped_series.sum()
        if total > 0:
            normalized = grouped_series / total
        else:
            normalized = grouped_series
        
        return normalized.sort_values(ascending=False).round(4)
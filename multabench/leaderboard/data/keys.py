# Column keys
MODEL = "model"
DATASET = "dataset"
TASK = "task"
DATASET_SIZE = "dataset_size"
FOLD = "fold"
IS_TUNED = "is_tuned"
TEST_SCORE = "test_score"
NORM_SCORE = "norm_score"
BEST_VAL_LOSS = "best_val_loss"
MM = "multimodal_state"
DINO_TUNE = "tune_dino"
E5_TUNE = "tune_e5"
E5_MODEL = "e5_model"
DINO_MODEL = "dino_model"

E5_SMALL = "e5-small"
E5_LARGE = "e5-large"
DINO_SMALL = "dino-small"
DINO_LARGE = "dino-large"

# E5 comparison mode labels
E5_SMALL_ALL = "E5-small (all)"
E5_SMALL_FT  = "E5-small (ft)"
E5_LARGE_ALL = "E5-large (all)"
E5_LARGE_FT  = "E5-large (ft)"

# DINO comparison mode labels
DINO_SMALL_ALL = "DINO-small (all)"
DINO_SMALL_FT  = "DINO-small (ft)"
DINO_LARGE_ALL = "DINO-large (all)"
DINO_LARGE_FT  = "DINO-large (ft)"

# Modality filter values
MODALITY = "modality"
MODALITY_TEXT = "Text"
MODALITY_IMAGE = "Image"

# Task type filter values
TASK_TYPE = "task_type"
TASK_REG = "Regression"
TASK_CLS = "Classification"

PCA_COMPONENTS = "pca_components"

# Display mode labels
MODE = "mode"
ALL_FEAT = "All"
FINETUNED = "Finetuned"
PURE_TABULAR = "Tabular Only"
IMG_ONLY = "Image Only"
NO_IMG = "No Image"
TEXT_ONLY = "Text Only"
NO_TEXT = "No Text"
FT_IMG_FT_TXT = "FT Img+Txt"
FT_TXT = "FT Text"

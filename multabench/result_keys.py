"""Keys of a run's result, shared by the benchmark runner and the leaderboard."""

from enum import StrEnum


class ResultKey(StrEnum):
    """A key of a run's result JSON, and a column of the result CSVs."""
    MODEL = "model"
    DATASET = "dataset"
    TEXT_ENCODER = "text_encoder"
    IMAGE_ENCODER = "image_encoder"
    SIZE = "size"
    FOLD = "fold"
    STATUS = "status"
    ERROR = "error"
    TRACEBACK = "traceback"
    METRIC = "metric"
    TEST_SCORE = "test_score"
    TEST_ERROR = "test_error"
    TRAIN_TIME_PER_1K = "train_time_per_1k_s"
    INFERENCE_TIME_PER_1K = "inference_time_per_1k_s"
    TRAIN_EMBEDDING_TIME_PER_1K = "train_embedding_time_per_1k_s"
    INFERENCE_EMBEDDING_TIME_PER_1K = "inference_embedding_time_per_1k_s"
    RUNTIME = "runtime"
    TASK_TYPE = "task_type"
    TRAIN_TYPE = "train_type"
    N_TRAIN = "n_train"
    N_TEST = "n_test"
    M_FEATURES = "m_features"
    TUNE_DINO = "tune_dino"
    TUNE_E5 = "tune_e5"
    E5_MODEL_NAME = "e5_model_name"
    BEST_VAL_LOSS = "best_val_loss"
    USE_GPU = "use_gpu"
    GPU_TYPE = "gpu_type"
    GPU_TOTAL_MEMORY_GB = "gpu_total_memory_gb"
    CPU_NAME = "cpu_name"
    CPU_CORES = "cpu_cores"
    SYSTEM_RAM_GB = "system_ram_gb"
    VISIBLE_CPUS = "visible_cpus"
    RAM_LIMIT_GB = "ram_limit_gb"
    GIT = "git"
    TIMESTAMP = "timestamp"


RUN_KEY = [ResultKey.MODEL, ResultKey.DATASET, ResultKey.TEXT_ENCODER, ResultKey.IMAGE_ENCODER, ResultKey.SIZE,
           ResultKey.FOLD]


class RunStatus(StrEnum):
    OK = "ok"
    ERROR = "error"

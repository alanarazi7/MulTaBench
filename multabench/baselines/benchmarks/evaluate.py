import time
from typing import Type, Dict

import torch

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.objects import MultimodalDataset
from multabench.baselines.abstract_model import TabularModel
from multabench.benchmark.load import load_multabench_dataset, split_dataset
from multabench.benchmark.splits import Size
from multabench.utils.hardware import get_hardware_dict
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.utils.logging import get_current_commit_hash
from multabench.result_keys import ResultKey
from multabench.utils.profiling import PeakMemoryTracker
from multabench.utils.timing import pop_embedding_seconds

def evaluate_on_loaded_dataset(model_cls: Type[TabularModel],
                                dataset: MultimodalDataset,
                                fold: int,
                                device: torch.device,
                                size: Size = Size.TEN_K,
                                verbose: bool = False,
                                text_encoder: TextEncoder | None = None,
                                image_encoder: ImageEncoder | None = None) -> Dict:
    start_time = time.time()
    dataset_id = dataset.dataset_id
    split = split_dataset(dataset, fold=fold, size=size)
    x_train, y_train = split.x_train, split.y_train
    x_test, y_test = split.x_test, split.y_test
    model = model_cls(problem_type=dataset.task_type, device=device, verbose=verbose, dataset=dataset_id,
                      image_folder=dataset.image_folder, image_column=dataset.image_column, text_encoder=text_encoder,
                      image_encoder=image_encoder, fold=fold, size=size)
    pop_embedding_seconds()
    with PeakMemoryTracker(phase='train', device=device) as train_tracker:
        model.fit(x_train, y_train)
    train_embedding_s = pop_embedding_seconds()
    with PeakMemoryTracker(phase='inference', device=device) as test_tracker:
        metrics = model.score_all_metrics(X=x_test, y=y_test)
    inference_embedding_s = pop_embedding_seconds()
    runtime = time.time() - start_time
    d_summary = {
        ResultKey.TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        ResultKey.GIT: get_current_commit_hash(),
        ResultKey.MODEL: model_cls.MODEL_NAME,
        ResultKey.DATASET: dataset_id.name,
        ResultKey.TASK_TYPE: str(dataset_id.name)[:3],
        ResultKey.TRAIN_TYPE: "benchmark",
        ResultKey.FOLD: fold,
        ResultKey.SIZE: size,
        **metrics,
        ResultKey.RUNTIME: runtime,
        ResultKey.N_TRAIN: len(y_train),
        ResultKey.N_TEST: len(y_test),
        ResultKey.M_FEATURES: x_train.shape[1],
        ResultKey.TUNE_DINO: bool(image_encoder and IMAGE_ENCODERS[image_encoder].tune_encoder),
        ResultKey.TUNE_E5: bool(text_encoder and TEXT_ENCODERS[text_encoder].tune_encoder),
        ResultKey.E5_MODEL_NAME: TEXT_ENCODERS[text_encoder].encoder_name if text_encoder else None,
        ResultKey.BEST_VAL_LOSS: getattr(model, "best_val_loss", None),
        **train_tracker.summary(),
        **test_tracker.summary(),
        ResultKey.TRAIN_TIME_PER_1K: (train_tracker.wall_time_s - train_embedding_s) / len(y_train) * 1000,
        ResultKey.INFERENCE_TIME_PER_1K: (test_tracker.wall_time_s - inference_embedding_s) / len(y_test) * 1000,
        ResultKey.TRAIN_EMBEDDING_TIME_PER_1K: train_embedding_s / len(y_train) * 1000,
        ResultKey.INFERENCE_EMBEDDING_TIME_PER_1K: inference_embedding_s / len(y_test) * 1000,
        **get_hardware_dict(device),
    }
    metric, error = metrics[ResultKey.METRIC], metrics[ResultKey.TEST_ERROR]
    print(f"{dataset_id.name} fold {fold}: {metric} error {error:.4f} ({runtime:.0f}s)")
    return d_summary


def evaluate_on_dataset(model_cls: Type[TabularModel],
                        dataset_id: MulTaBenchDatasetID,
                        fold: int,
                        device: torch.device,
                        size: Size = Size.TEN_K,
                        verbose: bool = False,
                        text_encoder: TextEncoder | None = None,
                        image_encoder: ImageEncoder | None = None) -> Dict:
    print(f"Running model {model_cls.MODEL_NAME} over dataset {dataset_id} with fold {fold}")
    dataset = load_multabench_dataset(dataset_id)
    return evaluate_on_loaded_dataset(
        model_cls=model_cls,
        dataset=dataset,
        fold=fold,
        device=device,
        size=size,
        verbose=verbose,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
    )

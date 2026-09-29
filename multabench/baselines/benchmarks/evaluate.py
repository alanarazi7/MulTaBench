import time
from typing import Type, Dict

import torch

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.objects import MultimodalDataset
from multabench.baselines.abstract_model import TabularModel
from multabench.benchmark.load import load_multabench_dataset
from multabench.benchmark.splits import get_split
from multabench.utils.hardware import get_hardware_dict
from multabench.dino.constants import DINOV3_SMALL
from multabench.e5.constants import E5_SMALL_V2
from multabench.utils.logging import get_current_commit_hash
from multabench.result_keys import METRIC, TEST_ERROR
from multabench.utils.profiling import PeakMemoryTracker

def evaluate_on_loaded_dataset(model_cls: Type[TabularModel],
                                dataset: MultimodalDataset,
                                fold: int,
                                device: torch.device,
                                verbose: bool = False,
                                tune_dino: bool = False,
                                dino_train_kwargs: dict | None = None,
                                dino_model_name: str = DINOV3_SMALL,
                                tune_e5: bool = False,
                                e5_train_kwargs: dict | None = None,
                                e5_model_name: str = E5_SMALL_V2) -> Dict:
    start_time = time.time()
    dataset_id = dataset.dataset_id
    train_idx, test_idx = get_split(dataset.y.to_numpy(), is_cls=dataset.is_cls, split=fold)
    x_train, y_train = dataset.x.iloc[train_idx], dataset.y.iloc[train_idx]
    x_test, y_test = dataset.x.iloc[test_idx], dataset.y.iloc[test_idx]
    kwargs = dict(problem_type=dataset.task_type, device=device, verbose=verbose, dataset=dataset_id,
                  image_folder=dataset.image_folder, tune_dino=tune_dino, dino_train_kwargs=dino_train_kwargs,
                  dino_model_name=dino_model_name,
                  tune_e5=tune_e5, e5_train_kwargs=e5_train_kwargs, e5_model_name=e5_model_name)
    model = model_cls(**kwargs)
    with PeakMemoryTracker(phase='train', device=device) as train_tracker:
        model.fit(x_train, y_train)
    with PeakMemoryTracker(phase='inference', device=device) as test_tracker:
        metrics = model.score_all_metrics(X=x_test, y=y_test)
    runtime = time.time() - start_time
    d_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "git": get_current_commit_hash(),
        "model": model_cls.MODEL_NAME,
        "dataset": dataset_id.name,
        "task_type": str(dataset_id.name)[:3],
        "train_type": "benchmark",
        "fold": fold,
        **metrics,
        "runtime": runtime,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "m_features": x_train.shape[1],
        'tune_dino': tune_dino,
        'tune_e5': tune_e5,
        'e5_model_name': e5_model_name,
        "best_val_loss": getattr(model, "best_val_loss", None),
        **train_tracker.summary(),
        **test_tracker.summary(),
        **get_hardware_dict(device),
        **(dino_train_kwargs or {}),
        **(e5_train_kwargs or {}),
    }
    print(f"{dataset_id.name} fold {fold}: {metrics[METRIC]} error {metrics[TEST_ERROR]:.4f} ({runtime:.0f}s)")
    return d_summary


def evaluate_on_dataset(model_cls: Type[TabularModel],
                        dataset_id: MulTaBenchDatasetID,
                        fold: int,
                        device: torch.device,
                        verbose: bool = False,
                        tune_dino: bool = False,
                        dino_train_kwargs: dict | None = None,
                        dino_model_name: str | None = None,
                        tune_e5: bool = False,
                        e5_train_kwargs: dict | None = None,
                        e5_model_name: str = E5_SMALL_V2) -> Dict:
    print(f"Running model {model_cls.MODEL_NAME} over dataset {dataset_id} with fold {fold}")
    dataset = load_multabench_dataset(dataset_id)
    return evaluate_on_loaded_dataset(
        model_cls=model_cls,
        dataset=dataset,
        fold=fold,
        device=device,
        verbose=verbose,
        tune_dino=tune_dino,
        dino_train_kwargs=dino_train_kwargs,
        dino_model_name=dino_model_name,
        tune_e5=tune_e5,
        e5_train_kwargs=e5_train_kwargs,
        e5_model_name=e5_model_name,
    )

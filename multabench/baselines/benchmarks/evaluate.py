import time
from dataclasses import asdict
from typing import Type, Dict

import torch

from tabstar.preprocessing.splits import split_to_test
from multabench.datasets.all_datasets import MultimodalDatasetID
from multabench.datasets.curation import MultimodalDataset
from multabench.baselines.abstract_model import TabularModel
from multabench.datasets.downloading import download_dataset
from multabench.datasets.multimodal import MultimodalState
from multabench.baselines.preprocessing.sampling import subsample_dataset
from multabench.utils.hardware import get_hardware_dict
from multabench.dino.constants import DINOV3_SMALL
from multabench.e5.constants import E5_SMALL_V2
from multabench.utils.logging import get_current_commit_hash
from multabench.utils.profiling import PeakMemoryTracker

DOWNSTREAM_EXAMPLES = 10_000
FOLDS = 10
MEMORY = "32G"


def evaluate_on_loaded_dataset(model_cls: Type[TabularModel],
                                dataset: MultimodalDataset,
                                fold: int,
                                device: torch.device,
                                train_examples: int = DOWNSTREAM_EXAMPLES,
                                verbose: bool = False,
                                memory: str = MEMORY,
                                multimodal_state: MultimodalState | None = None,
                                tune_dino: bool = False,
                                dino_train_kwargs: dict | None = None,
                                dino_model_name: str = DINOV3_SMALL,
                                tune_e5: bool = False,
                                e5_train_kwargs: dict | None = None,
                                e5_model_name: str = E5_SMALL_V2,
                                target_override: str | None = None,
                                pca_components: int = 30,
                                no_pca: bool = False) -> Dict:
    start_time = time.time()
    dataset_id = dataset.dataset_id
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, fold=fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=fold, train_examples=train_examples)
    kwargs = dict(problem_type=dataset.task_type, device=device, verbose=verbose, dataset=dataset_id,
                  image_folder=dataset.image_folder, tune_dino=tune_dino, dino_train_kwargs=dino_train_kwargs,
                  dino_model_name=dino_model_name,
                  tune_e5=tune_e5, e5_train_kwargs=e5_train_kwargs, e5_model_name=e5_model_name,
                  pca_components=pca_components, no_pca=no_pca)
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
        "train_examples": train_examples,
        "test_score": metrics.score,
        "metrics_dict": asdict(metrics),
        "runtime": runtime,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "m_features": x_train.shape[1],
        "request_memory": memory,
        'tune_dino': tune_dino,
        'tune_e5': tune_e5,
        'e5_model_name': e5_model_name,
        'multimodal_state': multimodal_state,
        'target_override': target_override,
        'pca_components': pca_components,
        'no_pca': no_pca,
        "best_val_loss": getattr(model, "best_val_loss", None),
        **train_tracker.summary(),
        **test_tracker.summary(),
        **get_hardware_dict(device),
        **(dino_train_kwargs or {}),
        **(e5_train_kwargs or {}),
    }
    print(f"Scored {metrics.score:.4f} on dataset {dataset_id.name}, fold {fold} in {int(runtime)} seconds. Multimodal state: {multimodal_state}")
    return d_summary


def evaluate_on_dataset(model_cls: Type[TabularModel],
                        dataset_id: MultimodalDatasetID,
                        fold: int,
                        device: torch.device,
                        train_examples: int = DOWNSTREAM_EXAMPLES,
                        verbose: bool = False,
                        memory: str = MEMORY,
                        multimodal_state: MultimodalState | None = None,
                        tune_dino: bool = False,
                        dino_train_kwargs: dict | None = None,
                        dino_model_name: str | None = None,
                        tune_e5: bool = False,
                        e5_train_kwargs: dict | None = None,
                        e5_model_name: str = E5_SMALL_V2,
                        target_override: str | None = None,
                        pca_components: int = 30,
                        no_pca: bool = False) -> Dict:
    print(f"Running model {model_cls.MODEL_NAME} over dataset {dataset_id} with fold {fold}")
    dataset = download_dataset(dataset_id=dataset_id, multimodal_state=multimodal_state, target_override=target_override)
    return evaluate_on_loaded_dataset(
        model_cls=model_cls,
        dataset=dataset,
        fold=fold,
        device=device,
        train_examples=train_examples,
        verbose=verbose,
        memory=memory,
        multimodal_state=multimodal_state,
        tune_dino=tune_dino,
        dino_train_kwargs=dino_train_kwargs,
        dino_model_name=dino_model_name,
        tune_e5=tune_e5,
        e5_train_kwargs=e5_train_kwargs,
        e5_model_name=e5_model_name,
        target_override=target_override,
        pca_components=pca_components,
        no_pca=no_pca,
    )

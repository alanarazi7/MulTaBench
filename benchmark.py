import argparse
import json
import os
import sys
import time
import traceback

from multabench.utils.devices import get_device
from multabench.baselines.autogluon_mm import AutoGluonMM
from multabench.baselines.catboost import CatBoost
from multabench.baselines.causilo import Causilo
from multabench.baselines.contexttab import ConTextTab
from multabench.baselines.kumo_tabular import KumoTabular, KumoTabularMedium, KumoTabularSmall
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.random_forest import RandomForest
from multabench.baselines.realmlp import RealMLP
from multabench.baselines.tabdpt import TabDPT
from multabench.baselines.tabfm import TabFM
from multabench.baselines.tabicl_v2 import TabICLv2
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfn import TabPFN3, TabPFN3p5, TabPFN3p5Fast, TabPFNv2, TabPFNv2p5
from multabench.baselines.tabstar_v1 import TabSTAR
from multabench.baselines.xgboost import XGBoost
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset
from multabench.benchmark.runs import Run
from multabench.benchmark.splits import SPLITS, Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.embeddings.hub import reads_cache
from multabench.result_keys import ResultKey, RunStatus
from multabench.utils.hardware import CPU_RUN_HARDWARE, assert_hardware, run_hardware
from multabench.utils.logging import assert_official_commit, get_current_commit_hash

BASELINES = [TabSTAR,
             CatBoost, XGBoost, LightGBM, RandomForest,
             RealMLP, TabM,
             TabICLv2, TabDPT,
             AutoGluonMM,
             ConTextTab,
             TabPFNv2, TabPFNv2p5,
             TabPFN3, TabPFN3p5, TabPFN3p5Fast,
             KumoTabular, KumoTabularMedium, KumoTabularSmall,
             Causilo, TabFM,]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), required=True)
    parser.add_argument('--dataset_name', type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    parser.add_argument('--fold', type=int, required=True, choices=range(SPLITS))
    parser.add_argument('--size', type=Size, default=Size.TEN_K, choices=list(Size))
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--overwrite', action='store_true', default=False, help="rerun even if the result JSON exists")
    parser.add_argument('--official', action='store_true', default=False,
                        help="raise unless on the leaderboard's hardware and a clean commit on master")
    parser.add_argument('--device', type=str, default=None, help="e.g. cuda:1; default: cuda, or cpu for runs that don't need a GPU")
    parser.add_argument('--image_encoder', type=ImageEncoder, default=None, choices=list(ImageEncoder),
                        help="for datasets with images; default: dino-small")
    parser.add_argument('--text_encoder', type=TextEncoder, default=None, choices=list(TextEncoder),
                        help="for datasets with text columns; default: e5-small")
    args = parser.parse_args()

    model = SHORT2MODELS[args.model]
    dataset = MulTaBenchDatasetID[args.dataset_name]
    modality = dataset_modality(dataset)
    if modality.has_text and model.USE_TEXT_EMBEDDINGS:
        args.text_encoder = args.text_encoder or TextEncoder.E5_SMALL
    elif args.text_encoder and modality.has_text:
        parser.error(f"{model.MODEL_NAME} reads raw text, so --text_encoder doesn't apply")
    elif args.text_encoder:
        parser.error(f"{dataset.name} has no text columns, so --text_encoder doesn't apply")
    if modality.has_images:
        args.image_encoder = args.image_encoder or ImageEncoder.DINO_SMALL
    elif args.image_encoder:
        parser.error(f"{dataset.name} has no images, so --image_encoder doesn't apply")
    image_encoder = IMAGE_ENCODERS[args.image_encoder] if args.image_encoder else None
    text_encoder = TEXT_ENCODERS[args.text_encoder] if args.text_encoder else None
    live_encoders = [e for e in (text_encoder, image_encoder) if e and not reads_cache(e)]
    hardware = run_hardware(model.NEEDS_GPU, encoders=live_encoders)
    device = get_device(device=args.device or ("cpu" if hardware == CPU_RUN_HARDWARE else None))
    if args.official:
        assert_official_commit()
        assert_hardware(device, hardware)
    run = Run(args.model, dataset.name, args.text_encoder, args.image_encoder, args.size, args.fold)
    out_path = os.path.join(args.output_dir, f"{run.name}.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Skipping: {out_path} exists (--overwrite to rerun)")
        sys.exit(0)
    try:
        ret = evaluate_on_dataset(
            model_cls=model,
            dataset_id=dataset,
            fold=args.fold,
            size=args.size,
            device=device,
            verbose=args.verbose,
            text_encoder=args.text_encoder,
            image_encoder=args.image_encoder,
        )
        ret[ResultKey.STATUS] = RunStatus.OK
    except Exception as e:
        traceback.print_exc()
        ret = {
            ResultKey.TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            ResultKey.GIT: get_current_commit_hash(),
            ResultKey.MODEL: model.MODEL_NAME,
            ResultKey.DATASET: dataset.name,
            ResultKey.FOLD: args.fold,
            ResultKey.SIZE: args.size,
            ResultKey.STATUS: RunStatus.ERROR,
            ResultKey.ERROR: f"{type(e).__name__}: {e}",
            ResultKey.TRACEBACK: traceback.format_exc(),
        }
    ret[ResultKey.TEXT_ENCODER] = args.text_encoder
    ret[ResultKey.IMAGE_ENCODER] = args.image_encoder
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ret, f, indent=2, default=str)
    print(f"Summary written to {out_path}: {ret}")
    if ret[ResultKey.STATUS] == RunStatus.ERROR:
        sys.exit(1)

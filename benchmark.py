import argparse
import json
import os
import sys
import time
import traceback

from multabench.utils.devices import get_device
from multabench.finetune.train_args import DinoTrainArgs, E5TrainArgs
from multabench.baselines.autogluon_mm import AutoGluonMM
from multabench.baselines.catboost import CatBoost
from multabench.baselines.contexttab import ConTextTab
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.random_forest import RandomForest
from multabench.baselines.realmlp import RealMLP
from multabench.baselines.tabdpt import TabDPT
from multabench.baselines.tabicl_v2 import TabICLv2
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.baselines.tabstar_v1 import TabSTAR
from multabench.baselines.xgboost import XGBoost
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset
from multabench.benchmark.splits import SIZE_10K, SIZES, SPLITS
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.dino.constants import IMAGE_ENCODERS
from multabench.e5.constants import TEXT_ENCODERS
from multabench.result_keys import STATUS, RunStatus
from multabench.utils.logging import get_current_commit_hash

BASELINES = [TabSTAR,
             CatBoost, XGBoost, LightGBM, RandomForest,
             RealMLP, TabM,
             TabICLv2, TabDPT,
             AutoGluonMM,
             ConTextTab,
             TabPFNv2, TabPFNv2p5,]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), required=True)
    parser.add_argument('--dataset_name', type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    parser.add_argument('--fold', type=int, required=True, choices=range(SPLITS))
    parser.add_argument('--size', type=str, default=SIZE_10K, choices=SIZES)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--overwrite', action='store_true', default=False, help="rerun even if the result JSON exists")
    parser.add_argument('--device', type=str, default=None, help="e.g. cuda:1 or cpu; default: cuda, then mps, then cpu")
    _dino = DinoTrainArgs()
    _e5 = E5TrainArgs()
    parser.add_argument('--image_encoder', type=str, default='dino-small', choices=list(IMAGE_ENCODERS))
    # DINO LoRA finetuning params (used by the "-tar" image encoders)
    parser.add_argument('--dino_lr', type=float, default=_dino.learning_rate)
    parser.add_argument('--dino_rank', type=int, default=_dino.lora_rank)
    parser.add_argument('--dino_img_layers', type=int, default=_dino.img_layers)
    parser.add_argument('--dino_epochs', type=int, default=_dino.epochs)
    parser.add_argument('--dino_patience', type=int, default=_dino.patience)
    parser.add_argument('--dino_weight_decay', type=float, default=_dino.weight_decay)
    parser.add_argument('--dino_batch_size', type=int, default=_dino.batch_size)
    parser.add_argument('--text_encoder', type=str, default='e5-small', choices=list(TEXT_ENCODERS))
    # E5 LoRA finetuning params (used by the "-tar" text encoders)
    parser.add_argument('--e5_lr', type=float, default=_e5.learning_rate)
    parser.add_argument('--e5_rank', type=int, default=_e5.lora_rank)
    parser.add_argument('--e5_text_layers', type=int, default=_e5.text_layers)
    parser.add_argument('--e5_epochs', type=int, default=_e5.epochs)
    parser.add_argument('--e5_patience', type=int, default=_e5.patience)
    parser.add_argument('--e5_weight_decay', type=float, default=_e5.weight_decay)
    parser.add_argument('--e5_batch_size', type=int, default=_e5.batch_size)
    args = parser.parse_args()

    model = SHORT2MODELS[args.model]
    dataset = MulTaBenchDatasetID[args.dataset_name]
    image_encoder = IMAGE_ENCODERS[args.image_encoder]
    text_encoder = TEXT_ENCODERS[args.text_encoder]
    device = get_device(device=args.device)
    exp_name = f"{args.model}_{dataset.name}_{args.text_encoder}_{args.image_encoder}_{args.size}_{args.fold}"
    out_path = os.path.join(args.output_dir, f"{exp_name}.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Skipping {exp_name}: {out_path} exists (--overwrite to rerun)")
        sys.exit(0)
    dino_train_kwargs = dict(
        lora_rank=args.dino_rank,
        img_layers=args.dino_img_layers,
        learning_rate=args.dino_lr,
        epochs=args.dino_epochs,
        patience=args.dino_patience,
        weight_decay=args.dino_weight_decay,
        batch_size=args.dino_batch_size,
    ) if image_encoder.tune_encoder else None
    e5_train_kwargs = dict(
        lora_rank=args.e5_rank,
        text_layers=args.e5_text_layers,
        learning_rate=args.e5_lr,
        epochs=args.e5_epochs,
        patience=args.e5_patience,
        weight_decay=args.e5_weight_decay,
        batch_size=args.e5_batch_size,
    ) if text_encoder.tune_encoder else None
    try:
        ret = evaluate_on_dataset(
            model_cls=model,
            dataset_id=dataset,
            fold=args.fold,
            size=args.size,
            device=device,
            verbose=args.verbose,
            tune_dino=image_encoder.tune_encoder,
            dino_train_kwargs=dino_train_kwargs,
            dino_model_name=image_encoder.encoder_name,
            tune_e5=text_encoder.tune_encoder,
            e5_train_kwargs=e5_train_kwargs,
            e5_model_name=text_encoder.encoder_name,
        )
        ret[STATUS] = RunStatus.OK
    except Exception as e:
        traceback.print_exc()
        ret = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "git": get_current_commit_hash(),
            "model": model.MODEL_NAME,
            "dataset": dataset.name,
            "fold": args.fold,
            "size": args.size,
            STATUS: RunStatus.ERROR,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    ret["text_encoder"] = args.text_encoder
    ret["image_encoder"] = args.image_encoder
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ret, f, indent=2, default=str)
    print(f"Summary written to {out_path}: {ret}")
    if ret[STATUS] == RunStatus.ERROR:
        sys.exit(1)

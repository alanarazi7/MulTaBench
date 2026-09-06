import argparse

from tabstar.training.devices import get_device

from multabench.datasets.multimodal import MultimodalError, MultimodalState
from multabench.datasets.utils import dataset_from_name
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
from multabench.baselines.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from multabench.constants import DEVICE
from multabench.datasets.all_datasets import MultimodalDatasetID, OpenMLDatasetID, is_image_dataset, is_text_dataset
from multabench.dino.constants import DINO_SMALL, DINO_LARGE, DINO_MODEL_NAMES
from multabench.e5.constants import E5_SMALL, E5_LARGE, E5_MODEL_NAMES, TF_IDF
from multabench.utils.logging import wandb_run, wandb_finish

BASELINES = [TabSTAR,
             CatBoost, XGBoost, LightGBM, RandomForest,
             RealMLP, TabM,
             TabICLv2, TabDPT,
             AutoGluonMM,
             ConTextTab,
             TabPFNv2, TabPFNv2p5,]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}


def is_invalid_model_dataset_pair(model_name: str, dataset_id: MultimodalDatasetID) -> bool:
    highly_multiclass = [OpenMLDatasetID.MUL_TEXT_FOOD_WINE_REVIEW]
    if model_name in {TabPFNv2.SHORT_NAME, TabPFNv2p5.SHORT_NAME}:
        if dataset_id in highly_multiclass:
            print(f"Skipping {dataset_id.name} for TabPFNv2/TabPFNv2p5 as it is a highly multiclass dataset.")
            return True
    return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--memory', type=str, default='32G')
    parser.add_argument('--multimodal_state', type=str,
                        choices=["all", "non", "ft", "img", "text_only", "no_text",
                                 "txt", "non_txt", "ft-txt", "ft-img-ft-txt"],
                        default="all")
    parser.add_argument('--target', type=str, default=None, help='Override target column. Append _discrete to discretize a numeric column into bins (multiclass).')
    parser.add_argument('--project', type=str, default='multimodal_benchmark_filtering_attempts_0224')
    _dino = DinoTrainArgs()
    _e5 = E5TrainArgs()
    # DINO image encoder model selection
    parser.add_argument('--dino_model', type=str, default=DINO_SMALL, choices=[DINO_SMALL, DINO_LARGE])
    # DINO LoRA finetuning params (used when --tune_dino)
    parser.add_argument('--tune_dino', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--dino_lr', type=float, default=_dino.learning_rate)
    parser.add_argument('--dino_rank', type=int, default=_dino.lora_rank)
    parser.add_argument('--dino_img_layers', type=int, default=_dino.img_layers)
    parser.add_argument('--dino_epochs', type=int, default=_dino.epochs)
    parser.add_argument('--dino_patience', type=int, default=_dino.patience)
    parser.add_argument('--dino_weight_decay', type=float, default=_dino.weight_decay)
    parser.add_argument('--dino_batch_size', type=int, default=_dino.batch_size)
    # E5 text encoder model selection
    parser.add_argument('--e5_model', type=str, default=E5_SMALL, choices=[E5_SMALL, E5_LARGE, TF_IDF])
    # E5 LoRA finetuning params (used when --tune_e5)
    parser.add_argument('--tune_e5', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--e5_lr', type=float, default=_e5.learning_rate)
    parser.add_argument('--e5_rank', type=int, default=_e5.lora_rank)
    parser.add_argument('--e5_text_layers', type=int, default=_e5.text_layers)
    parser.add_argument('--e5_epochs', type=int, default=_e5.epochs)
    parser.add_argument('--e5_patience', type=int, default=_e5.patience)
    parser.add_argument('--e5_weight_decay', type=float, default=_e5.weight_decay)
    parser.add_argument('--e5_batch_size', type=int, default=_e5.batch_size)
    parser.add_argument('--pca_components', type=int, default=30, help='Number of PCA components for image and text embeddings.')
    parser.add_argument('--no_pca', type=str, default='no', choices=['yes', 'no'],
                        help='Skip PCA and scaling for image/text embeddings. Exits early if dataset has >5 multimodal features.')
    args = parser.parse_args()
    states_mapping = {
        'all':           MultimodalState.ALL,
        'non':           MultimodalState.NON_IMAGE,
        'img':           MultimodalState.IMAGE_ONLY,
        'ft':            MultimodalState.ALL,
        'text_only':     MultimodalState.TEXT_ONLY,
        'no_text':       MultimodalState.NO_TEXT,
        'txt':           MultimodalState.TEXT_ONLY,   # text embeddings only, no FT
        'non_txt':       MultimodalState.NO_TEXT,     # tabular + image, no text, no FT
        'ft-txt':        MultimodalState.ALL,          # ALL + E5 FT
        'ft-img-ft-txt': MultimodalState.ALL,          # ALL + DINO FT + E5 FT (sequential)
    }
    mm_state = states_mapping.get(args.multimodal_state)

    no_pca = (args.no_pca == 'yes')
    model = SHORT2MODELS[args.model]
    dataset = dataset_from_name(name=args.dataset_name)
    args.tune_dino = (args.tune_dino == 'yes'
                      or (args.multimodal_state == "ft" and is_image_dataset(dataset))
                      or args.multimodal_state == "ft-img-ft-txt")
    args.tune_e5 = (args.tune_e5 == 'yes'
                    or (args.multimodal_state == "ft" and is_text_dataset(dataset))
                    or args.multimodal_state in {"ft-txt", "ft-img-ft-txt"})
    device = get_device(device=DEVICE)
    if is_invalid_model_dataset_pair(model_name=args.model, dataset_id=dataset):
        exit()
    exp_name = f"{args.model}_{dataset.name}_{args.multimodal_state}_{args.fold}"
    wandb_run(exp_name=exp_name, project=args.project)
    dino_train_kwargs = dict(
        lora_rank=args.dino_rank,
        img_layers=args.dino_img_layers,
        learning_rate=args.dino_lr,
        epochs=args.dino_epochs,
        patience=args.dino_patience,
        weight_decay=args.dino_weight_decay,
        batch_size=args.dino_batch_size,
    ) if args.tune_dino else None
    e5_train_kwargs = dict(
        lora_rank=args.e5_rank,
        text_layers=args.e5_text_layers,
        learning_rate=args.e5_lr,
        epochs=args.e5_epochs,
        patience=args.e5_patience,
        weight_decay=args.e5_weight_decay,
        batch_size=args.e5_batch_size,
    ) if args.tune_e5 else None
    try:
        ret = evaluate_on_dataset(
            model_cls=model,
            dataset_id=dataset,
            fold=args.fold,
            train_examples=args.train_examples,
            device=device,
            verbose=args.verbose,
            memory=args.memory,
            multimodal_state=mm_state,
            tune_dino=args.tune_dino,
            dino_train_kwargs=dino_train_kwargs,
            dino_model_name=DINO_MODEL_NAMES[args.dino_model],
            tune_e5=args.tune_e5,
            e5_train_kwargs=e5_train_kwargs,
            e5_model_name=E5_MODEL_NAMES.get(args.e5_model, args.e5_model),
            target_override=args.target,
            pca_components=args.pca_components,
            no_pca=no_pca,
        )
        wandb_finish(d_summary=ret)
    except MultimodalError as e:
        print(f"❌ {e} for dataset {dataset.name} and model {model.MODEL_NAME}")

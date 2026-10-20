from enum import StrEnum

from multabench.finetune.train_args import LoraArgs
from multabench.utils.encoders import Encoder

DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"



class ImageEncoder(StrEnum):
    DINO_SMALL = "dino-small"
    DINO_SMALL_TAR = "dino-small-tar"
    DINO_LARGE = "dino-large"


DEFAULT_IMAGE_ENCODER = ImageEncoder.DINO_SMALL


IMAGE_ENCODERS = {
    ImageEncoder.DINO_SMALL: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=False),
    ImageEncoder.DINO_SMALL_TAR: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=True),
    ImageEncoder.DINO_LARGE: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=False),
}

DINO_LORA = LoraArgs(learning_rate=1e-3, epochs=100)

# The modules LoRA adapts in each tuned ViT layer, layer.{i}: attention and MLP, as in E5.
LORA_IMAGE_TARGET_MODULES = [
    "attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
    "mlp.up_proj", "mlp.down_proj",
]

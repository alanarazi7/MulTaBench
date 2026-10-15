from enum import StrEnum

from multabench.utils.encoders import Encoder

DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class ImageEncoder(StrEnum):
    DINO_SMALL = "dino-small"
    DINO_SMALL_TAR = "dino-small-tar"
    DINO_LARGE = "dino-large"


IMAGE_ENCODERS = {
    ImageEncoder.DINO_SMALL: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=False),
    ImageEncoder.DINO_SMALL_TAR: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=True),
    ImageEncoder.DINO_LARGE: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=False),
}

# The modules LoRA adapts in each tuned ViT layer, layer.{i}: attention and MLP, as in E5. The paper's runs named the
# MLP's under attention, which matched nothing, so they adapted the attention projections alone.
LORA_IMAGE_TARGET_MODULES = [
    "attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
    "mlp.up_proj", "mlp.down_proj",
]

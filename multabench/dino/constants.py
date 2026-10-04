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

D_DINO_SMALL = 384
D_DINO_LARGE = 1024

DINO_SMALL_LAYERS = 12
DINO_LARGE_LAYERS = 24

DINO_DIM = {
    DINOV3_SMALL: D_DINO_SMALL,
    DINOV3_LARGE: D_DINO_LARGE,
}

DINO_NUM_LAYERS = {
    DINOV3_SMALL: DINO_SMALL_LAYERS,
    DINOV3_LARGE: DINO_LARGE_LAYERS,
}

LORA_IMAGE_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "mlp.up_proj", "mlp.down_proj",
]

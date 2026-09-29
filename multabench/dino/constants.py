DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# --image_encoder choices: name -> (encoder, fine-tune it on the task, i.e. target-aware "TAR")
IMAGE_ENCODERS = {
    "dino-small": (DINOV3_SMALL, False),
    "dino-small-tar": (DINOV3_SMALL, True),
    "dino-large": (DINOV3_LARGE, False),
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

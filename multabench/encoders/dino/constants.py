from multabench.finetune.train_args import LoraArgs

DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


DINO_LORA = LoraArgs(learning_rate=1e-3, epochs=100)

# The modules LoRA adapts in each tuned ViT layer, layer.{i}: attention and MLP, as in E5.
LORA_IMAGE_TARGET_MODULES = [
    "attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
    "mlp.up_proj", "mlp.down_proj",
]

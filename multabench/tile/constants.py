"""Constants for TILE (Lefebvre et al., 2026, https://github.com/soda-inria/tile), at the paper's default LLM."""

# Gated: its licence must be accepted on the Hugging Face account of HF_TOKEN.
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
LLAMA_3_1_8B_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
# The transformer blocks kept, from the paper's layer probing: the output of the 14th of 32.
TILE_LAYERS = 14
# Rows are batched by tokens, padding included, as in the paper.
TILE_MAX_TOKENS = 65536

import os
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoConfig


POSITIVE = 1
NEGATIVE = -1
STATES = [POSITIVE, NEGATIVE]

CHARACTERS = ["competitive", "difference_aversion", "self_interest", "social_welfare"]


MODEL_IDS = {
    "llama2_7b":        "meta-llama/Llama-2-7b-chat-hf",
    "llama3_8b":        "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5_7b":       "Qwen/Qwen2.5-7B-Instruct",
    "mistral_7b":       "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2_2b":        "google/gemma-2-2b-it",
    "gemma2_9b":        "google/gemma-2-9b-it",
}


@dataclass(frozen=True)
class ModelDims:
    """num_heads * head_dim is the o_proj INPUT width, which is NOT always hidden_dim:
    Gemma2 has a non-square o_proj (gemma2_2b 2048 in vs hidden 2304)."""
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_dim: int

    @property
    def head_width(self):
        return self.num_heads * self.head_dim


OFFLINE_DIMS = {
    "llama2_7b":    ModelDims(32, 32, 128, 4096),
    "llama3_8b":    ModelDims(32, 32, 128, 4096),
    "qwen2.5_7b":   ModelDims(28, 28, 128, 3584),
    "mistral_7b":   ModelDims(32, 32, 128, 4096),
    "gemma2_9b":    ModelDims(42, 16, 256, 3584),
    "gemma2_2b":    ModelDims(26, 8, 256, 2304),
}


@lru_cache(maxsize=None)
def resolve_dims(model):
    try:
        cfg = AutoConfig.from_pretrained(MODEL_IDS[model])
        num_heads = cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
        return ModelDims(cfg.num_hidden_layers, num_heads, head_dim, cfg.hidden_size)
    except Exception as exc:
        if model in OFFLINE_DIMS:
            print(f"[config] AutoConfig unavailable ({type(exc).__name__}); using offline dims for {model}")
            return OFFLINE_DIMS[model]
        raise


def deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

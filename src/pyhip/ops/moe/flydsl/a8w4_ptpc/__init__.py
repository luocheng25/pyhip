"""MoE A8W4（mxfp4 权重）：加载时 ``prepare_weights``，前向 ``fused_moe``；说明见 README.md。"""

from .moe import CONFIGS, decode_max_tokens, fused_moe, select_config
from .weights import MXFP4Weights, prepare_weights

__all__ = ["CONFIGS", "MXFP4Weights", "decode_max_tokens", "fused_moe", "prepare_weights", "select_config"]

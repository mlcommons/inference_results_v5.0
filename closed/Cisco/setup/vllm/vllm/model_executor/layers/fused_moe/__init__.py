from contextlib import contextmanager
from typing import Any, Dict, Optional

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.triton_utils import HAS_TRITON
import vllm.envs as envs

_config: Optional[Dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    import vllm.model_executor.layers.fused_moe.fused_marlin_moe  # noqa
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa
    import vllm.model_executor.layers.fused_moe.mlperf_fused_moe #noqa

    if envs.VLLM_MOE_MLPERF_KERNEL:
        from vllm.model_executor.layers.fused_moe.mlperf_fused_moe import (
            fused_experts, fused_moe, fused_topk, get_config_file_name,
            grouped_topk, invoke_fused_moe_kernel, moe_align_block_size)
    else:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts, fused_moe, fused_topk, get_config_file_name,
            grouped_topk, invoke_fused_moe_kernel, moe_align_block_size)

    __all__ += [
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
        "invoke_fused_moe_kernel",
        "moe_align_block_size",
    ]

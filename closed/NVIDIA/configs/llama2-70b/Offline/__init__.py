# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama2-70b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    enable_sort = False
    min_duration = 2400000


class HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = "fp8"
    vboost_slider = 1

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
    }


class BlackwellOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = 'fp4'

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'norm_quant_fusion': 'enable'
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1

    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 16.0
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x1_HighAccuracy(GH200_144GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x2(GH200_144GB_aarch64x1):
    system = KnownSystem.GH200_144GB_ARMx2
    offline_expected_qps = GH200_144GB_aarch64x1.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class GH200_144GB_aarch64x2_HighAccuracy(GH200_144GB_aarch64x2):
    pass

class H100_SXM_80GB_PP2x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    vboost_slider = 0

    gpu_batch_size = {'llama2-70b': 1024}
    offline_expected_qps = 75
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'reduce_fusion': 'enable',
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 1024,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_Triton_PP2x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2

    use_triton = True
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1

    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 75
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {'max_num_tokens': 1024}

@ConfigRegistry.register(
    HarnessType.Custom,
    AccuracyTarget.k_99_9,
    PowerSetting.MaxP
)
class H100_SXM_80GB_Custom_HighAccuracy(H100_SXM_80GB_Triton_PP2x1):
    system = KnownSystem.H100_SXM_80GBx1
    use_triton = True
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1
    gpu_batch_size = {'llama2-70b': 1024}
    offline_expected_qps = 75
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'reduce_fusion': 'enable',
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 1024,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(
    HarnessType.Triton,
    AccuracyTarget.k_99_9,      
    PowerSetting.MaxP
)
class H100_SXM_80GB_Triton_PP2x1_HighAccuracy(H100_SXM_80GB_Triton_PP2x1):
    pass
@ConfigRegistry.register(
    HarnessType.Custom,
    AccuracyTarget.k_99_9,
    PowerSetting.MaxP
)
class H100_SXM_80GB_Triton_PP2x1_HighAccuracy_CustomAlias(H100_SXM_80GB_Triton_PP2x1_HighAccuracy):
    pass

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_Triton_PP2x4(H100_SXM_80GB_Triton_PP2x1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 25 * 4
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_PP2x2(H100_SXM_80GB_PP2x1):
    system = KnownSystem.H100_SXM_80GBx4
    offline_expected_qps = H100_SXM_80GB_PP2x1.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_PP2x4(H100_SXM_80GB_PP2x2):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = H100_SXM_80GB_PP2x2.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ, "PP2")
class H100_SXM_80GB_MaxQ_PP2x4(H100_SXM_80GB_PP2x4):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 66
    power_limit = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_HighAccuracy_PP2x1(H100_SXM_80GB_PP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class H100_SXM_80GB_HighAccuracy_PP2x4(H100_SXM_80GB_PP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ, "PP2")
class H100_SXM_80GB_HighAccuracy_MaxQ_PP2x4(H100_SXM_80GB_MaxQ_PP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx2

    gpu_batch_size = {'llama2-70b': 1300}
    offline_expected_qps = 13

    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x2(H100_NVL_94GB_TP2x1):
    system = KnownSystem.H100_NVL_94GBx4
    offline_expected_qps = 10


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_TP2x4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.H100_NVL_94GBx8
    offline_expected_qps = 50
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_HighAccuracy_TP2x1(H100_NVL_94GB_TP2x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class H100_NVL_94GB_HighAccuracy_TP2x4(H100_NVL_94GB_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ, "TP2")
class H100_NVL_94GB_HighAccuracy_MaxQ_TP2x4(H100_NVL_94GB_MaxQ_TP2x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1

    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 14.4
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx1_MaxQ(H200_SXM_141GBx1):
    power_limit = 500
    offline_expected_qps = 12.4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H200_SXM_141GBx1_HighAccuracy_MaxQ(H200_SXM_141GBx1_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1_Triton(H200_SXM_141GBx1):
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy_Triton(H200_SXM_141GBx1_HighAccuracy):
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = H200_SXM_141GBx1.offline_expected_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx1_MaxQ):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = 86
    power_limit = 500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8_Triton(H200_SXM_141GBx8):
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy_Triton(H200_SXM_141GBx8):
    triton_num_clients_per_frontend = 1
    triton_num_frontends_per_model = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H200_SXM_141GBx8_HighAccuracy_MaxQ(H200_SXM_141GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx2(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP1")
class H200_SXM_141GBx2_TP2PP1(H200_SXM_141GBx2):
    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 24
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
        'gemm_swiglu_plugin': 'fp8'
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2PP1")
class H200_SXM_141GBx2_HighAccuracy_TP2PP1(H200_SXM_141GBx2_TP2PP1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'llama2-70b': 850}
    offline_expected_qps = 15
    trtllm_build_flags = {'max_num_tokens': 1024}
    trtllm_runtime_flags = {'max_num_tokens': 1024}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1_HighAccuracy(H200_SXM_141GB_CTSx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    gpu_batch_size = {'llama2-70b': 784}
    offline_expected_qps = 120


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8_HighAccuracy(H200_SXM_141GB_CTSx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx1(BlackwellOfflineGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx1
    vboost_slider = 1
    offline_expected_qps = 47
    gpu_batch_size = {'llama2-70b': 2048}
    trtllm_build_flags = {'max_num_tokens': 3584}
    trtllm_runtime_flags = {
        'max_num_tokens': 3584,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx1_HighAccuracy(B200_SXM_180GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx8(B200_SXM_180GBx1):
    system = KnownSystem.B200_SXM_180GBx8
    offline_expected_qps = B200_SXM_180GBx1.offline_expected_qps * 8

@ConfigRegistry.register(
    HarnessType.Triton,
    AccuracyTarget.k_99,
    PowerSetting.MaxP
)
class DGX_H100_H100_SXM_80GBx2_Triton(H100_SXM_80GB_Triton_PP2x1):
    system = KnownSystem.H100_SXM_80GBx2

@ConfigRegistry.register(
    HarnessType.Triton,
    AccuracyTarget.k_99_9,
PowerSetting.MaxP
)
class DGX_H100_H100_SXM_80GBx2_Triton_HA(DGX_H100_H100_SXM_80GBx2_Triton):
    pass

@ConfigRegistry.register(
    HarnessType.Triton,
    AccuracyTarget.k_99,
    PowerSetting.MaxP
)
class DGX_H100_H100_SXM_80GBx1_Triton(H100_SXM_80GB_Triton_PP2x1):
    system = KnownSystem.H100_SXM_80GBx1

@ConfigRegistry.register(
    HarnessType.Triton,
    AccuracyTarget.k_99_9,
    PowerSetting.MaxP
)
class DGX_H100_H100_SXM_80GBx1_Triton_HA(DGX_H100_H100_SXM_80GBx1_Triton):
    pass

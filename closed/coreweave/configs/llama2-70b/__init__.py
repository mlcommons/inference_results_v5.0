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

from code.common.constants import Benchmark
from configs.configuration import BenchmarkConfiguration


class GPUBaseConfig(BenchmarkConfiguration):
    benchmark = Benchmark.LLAMA2

    tensor_path = "build/preprocessed_data/llama2-70b/"
    llm_gen_config_path = "code/llama2-70b/tensorrt/generation_config.json"
    input_dtype = "int32"
    use_token_latencies = True
    use_graphs = False

    precision = "fp16"

    trtllm_build_flags = {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 2048,
        'max_input_len': 1024,
        'max_seq_len': 1024 + 1024,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
    }

    trtllm_runtime_flags = {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 2048,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': True,
    }

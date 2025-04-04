/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include<stdint.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

static __host__ __device__ inline int div_up(int m, int n) {
  return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES >
struct Uint_from_size_in_bytes {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<1> {
  using Type = uint8_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<4> {
  using Type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<8> {
  using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<16> {
  using Type = uint4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// develop
void fp32_to_fp16(__half* dst, const float* src, int count, cudaStream_t stream);

void fp16_to_fp32(float* dst, const __half* src, int count, cudaStream_t stream);

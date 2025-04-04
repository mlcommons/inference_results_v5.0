/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_fp16.h>
#include <stdint.h>

#include "NvInfer.h"

#ifndef DENSE_INPUT_TYPE
#define DENSE_INPUT_TYPE half // int8_t // float
#endif

void runGatherKernel(const DENSE_INPUT_TYPE** numericalInputPtrBuffer,
                     const int32_t** categoricalInputPtrBuffer,
                     const size_t* sampleSizesBuf,
                     const size_t* sampleOffsetsBuf,
                     DENSE_INPUT_TYPE* numericOutputBuf,
                     int32_t* categoricalOutputBuf,
                     int sampleCount,
                     int numericVolume,
                     int categoricalVolume,
                     cudaStream_t stream);

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

add_library(lwis
    src/lwis.cpp
)

set_target_properties(lwis PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(lwis
    glog
    nvinfer
    nvinfer_plugin
    ${CUDA_LIBRARIES}
    numa
)

target_include_directories(lwis
    PUBLIC
        ${LOADGEN_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
)

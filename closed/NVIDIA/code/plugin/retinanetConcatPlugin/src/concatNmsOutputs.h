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

#ifndef RETINANET_CONCAT_NMS_OUTPUTS_PLUGIN_H
#define RETINANET_CONCAT_NMS_OUTPUTS_PLUGIN_H

#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <cublasLt.h>


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>

#include <iostream>
#include <string>

namespace nvinfer1
{

namespace plugin
{

class RetinanetConcatNmsOutputsPlugin : public IPluginV2DynamicExt  {
public:
    RetinanetConcatNmsOutputsPlugin(const PluginFieldCollection *fc);

    // create the plugin at runtime from a byte stream
    RetinanetConcatNmsOutputsPlugin(const void* data, size_t length);
    ~RetinanetConcatNmsOutputsPlugin() noexcept override = default;

    // IPluginV2Ext fields
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void destroy() noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    virtual void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2Ext fields
    IPluginV2DynamicExt* clone() const noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    virtual int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T>
    void write(char*& buffer, const T& val) const;

    template <typename T>
    void read(const char*& buffer, T& val) const;

private:
    int mDevice;
    int mSMVersionMajor;
    int mSMVersionMinor;
    
    std::string mNamespace;
};

class RetinanetConcatNmsOutputsPluginCreator : public IPluginCreator {
public:
    RetinanetConcatNmsOutputsPluginCreator() = default;

    ~RetinanetConcatNmsOutputsPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // RETINANET_CONCAT_NMS_OUTPUTS_PLUGIN_H

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

#ifndef LWIS_BUFFERS_H
#define LWIS_BUFFERS_H

#include "NvInfer.h"
#include "half.h"
#include <NvInferRuntime.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace lwis
{
#define DEBUG_LWIS_BUFFER 0

/* read in engine file into character array */
inline size_t GetModelStream(std::vector<char>& dst, std::string const& engineName)
{
    size_t size{0};
    std::ifstream file(engineName, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        dst.resize(size);
        file.read(dst.data(), size);
        file.close();
    }

    return size;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    // Fall through to error
    default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline int64_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format, const bool hasImplicitBatch)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    int channelDim{-1};
    switch (format)
    {
    case nvinfer1::TensorFormat::kCHW2:
        spv = 2;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW4:
        spv = 4;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kDHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kCHW16:
        spv = 16;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW32:
        spv = 32;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCDHW32:
        spv = 32;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kDLA_HWC4:
        spv = 4;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kDLA_LINEAR:
        spv = 64;
        channelDim = d_new.nbDims - 1;
        break;
    case nvinfer1::TensorFormat::kLINEAR:
        spv = 1;
        channelDim = -1;
        break;
    default:
        throw std::runtime_error("Invalid TensorFormat.");
        break;
    }
    if (spv > 1)
    {
        assert(channelDim >= 0); // Make sure we have valid channel dimension.
        d_new.d[channelDim] = roundUp(d_new.d[channelDim], spv);
#if DEBUG_LWIS_BUFFER
        std::cout << "{";
        for (int32_t ind = 0; ind < d_new.nbDims; ++ind)
        {
            std::cout << d_new.d[ind] << ", ";
        }
        std::cout << "}" << std::endl;
#endif
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const bool hasImplicitBatch)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer()
        : mByteSize(0)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size)
        : mByteSize(size)
    {
        if (!allocFn(&mBuffer, mByteSize))
            throw std::bad_alloc();
    }

    GenericBuffer(GenericBuffer&& buf)
        : mByteSize(buf.mByteSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mByteSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mByteSize = buf.mByteSize;
            mBuffer = buf.mBuffer;
            buf.mByteSize = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const
    {
        return mByteSize;
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mByteSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMallocHost(ptr, size) == cudaSuccess;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFreeHost(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;
using EngineTensors = std::vector<void*>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
//!          We distinguish the concept of tensors and buffers as follows:
//!          Tensors: buffer pointers used for engine's enqueue each loop
//!          Buffers: total memory allocated for each engine's input and output tensors, with loop count considered.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engines.
    //!
    BufferManager(std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> const& engines,
        std::vector<size_t> const& batchSizes, std::vector<size_t> const& batchLoopCounts,
        std::vector<int32_t> const& profileIndices)
        : mEngines(engines)
        , mBatchSizes(batchSizes)
        , mBatchLoopCounts(batchLoopCounts)
        , mProfileIndices(profileIndices)
    {
        // Manage buffers for each engine
        mHostTensors.resize(engines.size());
        mDeviceTensors.resize(engines.size());
        mManagedBuffers.resize(engines.size());
        // mHostTensors structure:
        // [engine1: [loop1: [profile1 profile2 profile3]]]
        // profile1 = (tensor1, tensor2, ...)
        for (size_t engineIdx = 0; engineIdx < engines.size(); ++engineIdx)
        {
            // Each optimization profile owns numIOTensors tensors.
            std::shared_ptr<nvinfer1::ICudaEngine> engine = engines[engineIdx];
            size_t batchSize = batchSizes[engineIdx];
            size_t batchLoopCount = batchLoopCounts[engineIdx];
            int32_t profileIdx = profileIndices[engineIdx];

            int32_t numIOTensors = engine->getNbIOTensors();
            int32_t numTensorsTotal = numIOTensors * engine->getNbOptimizationProfiles();
            
            int32_t tensorOffset = profileIdx * numIOTensors;

            // Allocate batchLoopCount * numTensorsTotal TRT tensors for batchLoopCount times inference
            mHostTensors[engineIdx].resize(batchLoopCount);
            mDeviceTensors[engineIdx].resize(batchLoopCount);

            // Reserve re spaces for each enqueue
            for (EngineTensors& hostTensors : mHostTensors[engineIdx])
            {
                hostTensors.resize(numTensorsTotal, nullptr);
            }
            for (EngineTensors& deviceTensors : mDeviceTensors[engineIdx])
            {
                deviceTensors.resize(numTensorsTotal, nullptr);
            }

            for (size_t IOTensorIndex = 0; IOTensorIndex < numIOTensors; IOTensorIndex++)
            {
                auto tensorName{engine->getIOTensorName(IOTensorIndex)};
                // Create host and device buffers
                nvinfer1::TensorIOMode isInput = engine->getTensorIOMode(tensorName);

                size_t vol = lwis::volume(engine->getTensorShape(tensorName),
                    engine->getTensorFormat(tensorName, profileIdx), false);
                size_t elementSize = lwis::getElementSize(engine->getTensorDataType(tensorName));
                size_t tensorSize = batchSize * vol * elementSize;

                // The first engine will have both input and output buffer constructed. Every engine starts from the
                // second one (engineIdx >=1) will only have the output buffer constructed, while reusing the previous
                // engine's output buffer as the input buffer.
                if (engineIdx >= 1 && isInput == nvinfer1::TensorIOMode::kINPUT)
                {
                    // Start from the second engine, reuse the previous engine output buffer as current input buffer
#if DEBUG_LWIS_BUFFER
                    std::cout << "Engine-" << engineIdx << " reusing address from Engine-" << engineIdx - 1
                              << ": host address-" << mManagedBuffers[engineIdx - 1].at(tensorName)->hostBuffer.data()
                              << " device address-"
                              << mManagedBuffers[engineIdx - 1].at(tensorName)->deviceBuffer.data() << " | ";
                    std::cout << (isInput ? "Input" : "Output") << " tensorName: " << tensorName
                              << " TensorSize: " << tensorSize << std::endl;
#endif
                    for (size_t loop = 0; loop < batchLoopCount; ++loop)
                    {
                        uint8_t* packedHostTensor
                            = static_cast<uint8_t*>(mManagedBuffers[engineIdx - 1].at(tensorName)->hostBuffer.data())
                            + tensorSize * loop;
                        uint8_t* packedDeviceTensor
                            = static_cast<uint8_t*>(mManagedBuffers[engineIdx - 1].at(tensorName)->deviceBuffer.data())
                            + tensorSize * loop;
                        mHostTensors[engineIdx][loop][tensorOffset + IOTensorIndex]
                            = static_cast<void*>(packedHostTensor);
                        mDeviceTensors[engineIdx][loop][tensorOffset + IOTensorIndex]
                            = static_cast<void*>(packedDeviceTensor);
                    }
                }
                // Construct output buffers for every engine starts from the second one
                // Construct both input and output buffers for the first engine
                else
                {
                    // To loop each input batchLoopCount times, we allocate total memory of batchLoopCount *
                    // tensorBufferSize for that input
                    size_t allocationSize = batchLoopCount * tensorSize;
                    std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
                    manBuf->deviceBuffer = DeviceBuffer(allocationSize);
                    manBuf->hostBuffer = HostBuffer(allocationSize);
#if DEBUG_LWIS_BUFFER
                    std::cout << "Allocated Address: host address-" << manBuf->hostBuffer.data() << " device address-"
                              << manBuf->deviceBuffer.data() << " | ";
                    std::cout << (isInput ? "Input" : "Output") << " tensorName: " << tensorName
                              << " TensorSize: " << tensorSize << " AllocationSize: " << allocationSize << std::endl;
#endif
                    for (size_t loop = 0; loop < batchLoopCount; ++loop)
                    {
                        // Shift tensor memory pointer for each loop
                        uint8_t* packedHostTensor
                            = static_cast<uint8_t*>(manBuf->hostBuffer.data()) + tensorSize * loop;
                        uint8_t* packedDeviceTensor
                            = static_cast<uint8_t*>(manBuf->deviceBuffer.data()) + tensorSize * loop;
                        mHostTensors[engineIdx][loop][tensorOffset + IOTensorIndex]
                            = static_cast<void*>(packedHostTensor);
                        mDeviceTensors[engineIdx][loop][tensorOffset + IOTensorIndex]
                            = static_cast<void*>(packedDeviceTensor);
                    }
                    // Move to the output buffer of this engine
                    mManagedBuffers[engineIdx].emplace(tensorName, std::move(manBuf));
                }
            }
#if DEBUG_LWIS_BUFFER
            std::cout << "Engine-" << engineIdx << " mManagedBuffers size: " << mManagedBuffers[engineIdx].size()
                      << std::endl;
#endif
        }
    }

    //!
    //! \brief Returns a vector of device tensors for all the engines that you can use directly as
    //!        tensors for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<std::vector<EngineTensors>>& getDeviceTensors()
    {
        return mDeviceTensors;
    }

    //!
    //! \brief Returns a vector of device tensors for all the engines.
    //!
    std::vector<std::vector<EngineTensors>> const& getDeviceTensors() const
    {
        return mDeviceTensors;
    }

    //!
    //! \brief Returns a vector of host tensors for all the engines that you can use directly as
    //!        tensors for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<std::vector<EngineTensors>>& getHostTensors()
    {
        return mHostTensors;
    }

    //!
    //! \brief Returns a vector of host tensors for all the engines.
    //!
    std::vector<std::vector<EngineTensors>> const& getHostTensors() const
    {
        return mHostTensors;
    }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(size_t const engineIdx, std::string const& tensorName) const
    {
        return getBuffer(engineIdx, false, tensorName);
    }

    //!
    //! \brief Returns the device buffer corresponding to index.
    //!
    void* getDeviceBuffer(size_t const engineIdx, size_t const tensorIndex) const
    {
        return getBuffer(engineIdx, false, tensorIndex);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getHostBuffer(size_t const engineIdx, std::string const& tensorName) const
    {
        return getBuffer(engineIdx, true, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to index.
    //!
    void* getHostBuffer(size_t const engineIdx, size_t const index) const
    {
        return getBuffer(engineIdx, true, index);
    }

    //!
    //! \brief Returns the batch loop count for an engine.
    //!
    size_t getBatchLoopCount(size_t const engineIdx) const
    {
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        return mBatchLoopCounts[engineIdx];
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(size_t const engineIdx, std::string const& tensorName) const
    {
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        return mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.size();
    }

    //!
    //! \brief Returns the volume of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t volume(size_t const engineIdx, std::string const& tensorName) const
    {
        auto tensorName_s = tensorName.c_str();
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        return lwis::volume(
            mEngines[engineIdx]->getTensorShape(tensorName_s),
            mEngines[engineIdx]->getTensorFormat(tensorName_s, mProfileIndices[engineIdx]),
            false
        );
    }

    //!
    //! \brief Returns the elementSize of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t elementSize(size_t const engineIdx, const std::string& tensorName) const
    {
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        return lwis::getElementSize(mEngines[engineIdx]->getTensorDataType(tensorName.c_str()));
    }

    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void dumpBuffer(size_t const engineIdx, std::ostream& os, const std::string& tensorName)
    {
        auto tensorName_s = tensorName.c_str();
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        void* buf = mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.data();
        size_t bufSize = mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.size();
        nvinfer1::Dims bufDims = mEngines[engineIdx]->getTensorShape(tensorName_s);
        size_t rowCount
            = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSizes[engineIdx]);

        os << "[" << mBatchSizes[engineIdx];
        for (int i = 0; i < bufDims.nbDims; i++)
        {
            os << ", " << bufDims.d[i];
        }
        os << "]" << std::endl;
        switch (mEngines[engineIdx]->getTensorDataType(tensorName_s))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: print<int8_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kBOOL: assert(0 && "Bool network-level input and output is not supported"); break;
        }
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
            {
                os << typedBuf[i] << std::endl;
            }
            else if (rowCount == 1)
            {
                os << typedBuf[i];
            }
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
            {
                os << typedBuf[i];
            }
            else if (i % rowCount == rowCount - 1)
            {
                os << " " << typedBuf[i] << std::endl;
            }
            else
            {
                os << " " << typedBuf[i];
            }
        }
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(size_t const engineIdx, size_t const tensorIdx, cudaStream_t const& stream = 0, void* src = nullptr,
        size_t const size = 0, size_t const offset = 0)
    {
        memcpyBuffers(engineIdx, tensorIdx, true, false, true, stream, src, size, offset);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(size_t const engineIdx, size_t const tensorIdx, cudaStream_t const& stream = 0, void* dst = nullptr,
        size_t const size = 0, size_t const offset = 0)
    {
        memcpyBuffers(engineIdx, tensorIdx, false, true, true, stream, dst, size, offset);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(size_t const engineIdx, bool const isHost, std::string const& tensorName) const
    {
        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        return (isHost ? mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.data()
                       : mManagedBuffers[engineIdx].at(tensorName)->deviceBuffer.data());
    }

    void* getBuffer(size_t const engineIdx, bool const isHost, size_t const tensorIdx) const
    {
        CHECK(mEngines[engineIdx]->getIOTensorName(tensorIdx) != nullptr);
        std::string tensorName{mEngines[engineIdx]->getIOTensorName(tensorIdx)};
        return (isHost ? mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.data()
                       : mManagedBuffers[engineIdx].at(tensorName)->deviceBuffer.data());
    }

    void memcpyBuffers(size_t const engineIdx, size_t const tensorIdx, bool const copyInput, bool const deviceToHost,
        bool const async, cudaStream_t const& stream = 0, void* buf = nullptr, size_t const size = 0,
        size_t const offset = 0)
    {
        auto tensorName{mEngines[engineIdx]->getIOTensorName(tensorIdx)};

        CHECK(engineIdx < mEngines.size()) << "Invalid engine index.";
        CHECK((copyInput && mEngines[engineIdx]->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
            || (!copyInput && mEngines[engineIdx]->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT))
            << "Expecting tensor " << tensorIdx << " to be " << (copyInput ? "input" : "output")
            << "but get the opposite.";
        size_t numBufferTensors = mManagedBuffers[engineIdx].size();
        CHECK(mEngines[engineIdx]->getIOTensorName(tensorIdx) != nullptr);
        CHECK(mManagedBuffers[engineIdx].find(tensorName) != mManagedBuffers[engineIdx].end())
            << "Invalid tensorIdx: " << tensorIdx << " numBufferTensors: " << numBufferTensors
            << " from Engine: " << engineIdx << " copy input: " << copyInput;

        void* dstPtr = deviceToHost ? (buf ? buf : mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.data())
                                    : static_cast<char*>(mManagedBuffers[engineIdx].at(tensorName)->deviceBuffer.data())
                + (buf ? offset * size : 0);
        void const* srcPtr = deviceToHost
            ? static_cast<char*>(mManagedBuffers[engineIdx].at(tensorName)->deviceBuffer.data())
                + (buf ? 0 : offset * size)
            : (buf ? buf : mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.data());
        size_t const byteSize = buf && size ? size : mManagedBuffers[engineIdx].at(tensorName)->hostBuffer.size();
        cudaMemcpyKind const memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        if (async)
        {
            CHECK_EQ(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream), cudaSuccess);
        }
        else
        {
            CHECK_EQ(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType), cudaSuccess);
        }
    }

    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> mEngines; //!< The pointer to the engines
    // tensors of each engine
    std::vector<size_t> mBatchSizes;      //!< The batch sizes of each engine
    std::vector<size_t> mBatchLoopCounts; //!< The loop count of each engine
    std::vector<int32_t> mProfileIndices; //!< The optimization profile indices of each engine
    std::vector<std::unordered_map<std::string, std::unique_ptr<ManagedBuffer>>>
        mManagedBuffers; //!< The vector of pointers to managed buffers of each engine
    std::vector<std::vector<EngineTensors>>
        mHostTensors; //!< The vector of host buffers needed for each engine execution
    std::vector<std::vector<EngineTensors>>
        mDeviceTensors; //!< The vector of device buffers needed for each engine execution
};

} // namespace lwis

#endif // LWIS_BUFFERS_H

/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
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

/* Include necessary header files */
// TRT
#include "NvInferPlugin.h"
#include "logger.h"

// TRITON
#include "bert_concurrent_frontend.hpp"
#include "dla_concurrent_frontend.hpp"
#include "triton_concurrent_frontend.hpp"

// QSL
#include "callback.hpp"
#include "qsl.hpp"

// Google Logging
#include <gflags/gflags.h>
#include <glog/logging.h>

// LoadGen
#include "loadgen.h"

// General C++
#include <dlfcn.h>
#include <numeric>
#include <vector>

// NUMA utils
#include "utils.hpp"

#include "cuda_profiler_api.h"

/* Define the appropriate flags */
// General flags
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(use_graphs, false, "Enable cudaGraphs for TensorRT engines");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set. 0=use default");
DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");
DEFINE_string(response_postprocess, "", "Enable imagenet post-processing on query sample responses.");
DEFINE_string(numa_config, "",
    "NUMA settings: each NUMA node contains a pair "
    "of GPU indices and CPU indices.");

DEFINE_uint32(num_batchers, 2, "Use triton_server_v2 with concurrent batching and issue.");
DEFINE_uint32(num_issuers, 2, "Use triton_server_v2 with concurrent batching and issue.");
DEFINE_bool(is_bert_benchmark, false, "Harness is running bert");
DEFINE_bool(is_single_stream, false, "If running single stream in concurrent harness, setting this reduces latency");

// DLA harness flags
DEFINE_bool(use_dla, false, "Use triton dla harness with gpu harness to issue requests for both GPU and DLA");
DEFINE_uint32(dla_batch_size, 1, "Batch size for dla requests");
DEFINE_string(dla_model_name, "dla_model", "Triton model repo model name for dla requests");
DEFINE_uint32(dla_model_version, 1, "Version of the DLA model to use with TRITON");
DEFINE_uint32(dla_num_batchers, 2, "Number of threads to batch DLA requests. Used only with DLA harness");
DEFINE_uint32(
    dla_num_issuers, 2, "Number of threads to issue DLA model requests to triton. Used only with DLA harness ");
DEFINE_double(dla_request_ratio, 0.25,
    "Ratio of total samples that will be run on DLA vs total number of requests. Used "
    "only with DLA harness");

// TRITON flags
DEFINE_string(model_store_path, "", "Path to the engines directory for server scenario");
DEFINE_string(tensorrt_backend_path, "", "Path to the library with tensorrt Backend in triton");
DEFINE_string(model_name, "", "Name of the model to use with TRITON");
DEFINE_uint32(model_version, 1, "Version of the model to use with TRITON");
DEFINE_uint32(buffer_manager_thread_count, 0, "The number of buffer manager thread");
DEFINE_bool(pinned_input, true, "Start inference assuming the data is in pinned memory");

// QSL flags
DEFINE_uint32(batch_size, 1, "Max Batch size to use for all devices and engines");
DEFINE_bool(batch_triton_requests, false, "Request batch size for each triton request");
DEFINE_bool(check_contiguity, false, "Check if requests in a single IssueQuery are conitguous");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "", "Path to preprocessed samples in npy format (<full_image_name>.npy)");
DEFINE_bool(coalesced_tensor, false, "Turn on if all the samples are coalesced into one single npy file");
DEFINE_bool(start_from_device, false, "Start inference assuming the data is already in device memory");
DEFINE_bool(end_on_device, false, "Store inference outputs in device memory");

// Loadgen test settings
DEFINE_string(scenario, "Server", "Scenario to run for Loadgen (Offline, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "resnet50", "Model name");
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// FIXME: this is enabled for MIG, but not used for SingleMIG or non-MIG
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap
    = {{"SingleStream", mlperf::TestScenario::SingleStream}, {"MultiStream", mlperf::TestScenario::MultiStream},
        {"Server", mlperf::TestScenario::Server}, {"Offline", mlperf::TestScenario::Offline}};

triton_frontend::ServerSUTPtr_t getInitializedServerSUT()
{
    triton_frontend::ServerSUTPtr_t sut = nullptr;
    if (FLAGS_use_dla)
    {
        sut = std::make_shared<triton_frontend::DLA_Triton_SUT>("DLA_Triton_Server", FLAGS_model_store_path,
            FLAGS_model_name, FLAGS_model_version, FLAGS_dla_model_name, FLAGS_dla_model_version,
            FLAGS_start_from_device, FLAGS_pinned_input, FLAGS_request_pool_count, FLAGS_num_batchers,
            FLAGS_num_issuers, FLAGS_dla_batch_size, FLAGS_dla_num_batchers, FLAGS_dla_num_issuers,
            FLAGS_dla_request_ratio);
    }
    else if (FLAGS_is_bert_benchmark)
    {
        sut = std::make_shared<triton_frontend::BERT_Triton_SUT>("BERT_Triton_Server", FLAGS_model_store_path,
            FLAGS_model_name, FLAGS_model_version, FLAGS_start_from_device, FLAGS_pinned_input,
            FLAGS_request_pool_count, FLAGS_num_batchers, FLAGS_num_issuers, FLAGS_is_single_stream);
    }
    else
    {
        sut = std::make_shared<triton_frontend::Concurrent_Frontend_SUT>("Concurrent_Triton_Server",
            FLAGS_model_store_path, FLAGS_model_name, FLAGS_model_version, false, FLAGS_start_from_device,
            FLAGS_pinned_input, FLAGS_request_pool_count, FLAGS_num_batchers, FLAGS_num_issuers,
            FLAGS_is_single_stream);
    }

    size_t min_sample_size = 1;
    size_t max_sample_size = 1;

    sut->Init(min_sample_size, max_sample_size, FLAGS_buffer_manager_thread_count, FLAGS_batch_triton_requests,
        FLAGS_check_contiguity, FLAGS_numa_config, FLAGS_tensorrt_backend_path);

    return sut;
}

void doInference()
{
    // Configure the test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario, 2);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario, 1);
    test_settings.server_coalesce_queries = true;

    // Configure the logging settings
    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = FLAGS_logfile_outdir;
    log_settings.log_output.prefix = FLAGS_logfile_prefix;
    log_settings.log_output.suffix = FLAGS_logfile_suffix;
    log_settings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    log_settings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    log_settings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
    log_settings.log_mode = logModeMap[FLAGS_log_mode];
    log_settings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
    log_settings.enable_trace = FLAGS_log_enable_trace;

    // Instantiate our SUT and get the status of the server and model
    triton_frontend::ServerSUTPtr_t sut = getInitializedServerSUT();

    sut->ModelMetadata();
    sut->SetResponseCallback(callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
                                                                       // post-processing callback

    // Instantiate our QSL
    std::shared_ptr<mlperf::QuerySampleLibrary> lib;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);

    auto numaConfig = parseNumaConfig(FLAGS_numa_config);
    size_t padding = test_settings.scenario == mlperf::TestScenario::MultiStream
        ? test_settings.multi_stream_samples_per_query - 1
        : 0;
    if (numaConfig.empty())
    {
        auto qsl = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", FLAGS_map_path,
            splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding, FLAGS_coalesced_tensor,
            start_from_device);
        lib = qsl;
        sut->AddSampleLibrary(qsl);
    }
    else
    {
        std::vector<qsl::SampleLibraryPtr_t> qsls;
        const int32_t nbNumas = numaConfig.size();
        auto gpuNumaVec = getGpuToNumaMap(numaConfig);
        for (int32_t numaIdx : gpuNumaVec)
        {
            // Use a thread to construct QSL so that the allocated memory is closer to that NUMA
            // node.
            auto constructQsl = [&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                bindNumaMemPolicy(numaIdx, nbNumas);
                auto oneQsl = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", FLAGS_map_path,
                    splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding,
                    FLAGS_coalesced_tensor, start_from_device);
                resetNumaMemPolicy();
                qsls.emplace_back(oneQsl);
            };
            std::thread th(constructQsl);
            bindThreadToCpus(th, numaConfig[numaIdx].second);
            th.join();
            sut->AddSampleLibrary(qsls.back());
        }
        lib.reset(new qsl::SampleLibraryEnsemble(qsls));
    }

    // Warmup our SUT
    double expected_qps{1.0};
    switch (test_settings.scenario)
    {
    case mlperf::TestScenario::Offline: expected_qps = test_settings.offline_expected_qps; break;
    case mlperf::TestScenario::Server: expected_qps = test_settings.server_target_qps; break;
    case mlperf::TestScenario::MultiStream:
        expected_qps = 1.0e9 / test_settings.multi_stream_expected_latency_ns;
        break;;
    case mlperf::TestScenario::SingleStream:
        expected_qps = 1.0e9 / test_settings.single_stream_expected_latency_ns;
        break;
    }

    sut->Warmup(FLAGS_warmup_duration, expected_qps);
    LOG(INFO) << "Start running actual test" << std::endl;
    // Perform the inference testing
    cudaProfilerStart();
    mlperf::StartTest(sut.get(), lib.get(), test_settings, log_settings);
    cudaProfilerStop();

    // Check SUT end status and inform the SUT that we are done
    sut->ModelStats();
    sut->Done();

    // Reset the SUT and QSL pointers
    sut.reset();
    lib.reset();
}

int main(int argc, char** argv)
{
    // Initialize the NVInfer plugins
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Parse command line flags
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            std::cout << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    // Perform Inference
    doInference();

    /* Return pass */
    return 0;
}

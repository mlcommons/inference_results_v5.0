The efficientnet models are run using the [TFLite CPP implementation](https://github.com/mlcommons/mlperf-automations/tree/main/script/app-mlperf-inference-tflite-cpp). Tensorflow lite is built from source with XNNPACK enabled. On arm64 platforms, ArmNN library is also used. 

Please follow the [MLPerf Inference docs](https://docs.mlcommons.org/inference/benchmarks/image_classification/mobilenets/#__tabbed_1_5) for the commands to run the models.

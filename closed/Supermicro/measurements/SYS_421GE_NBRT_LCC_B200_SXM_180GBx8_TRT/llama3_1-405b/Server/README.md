To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then to generate the TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=Server"
make run_harness RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=Server --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=llama3_1-405b --scenarios=Server --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`.
To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then to generate the TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=llama2-70b-interactive --scenarios=Offline"
make run_harness RUN_ARGS="--benchmarks=llama2-70b-interactive --scenarios=Offline --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=llama2-70b-interactive --scenarios=Offline --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`.
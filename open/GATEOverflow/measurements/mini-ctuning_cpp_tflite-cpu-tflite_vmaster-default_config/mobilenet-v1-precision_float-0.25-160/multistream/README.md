*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-53-generic-x86_64-with-glibc2.39
* CPU version: x86_64
* Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=e69f190613e627c084d4103da43bc1463c0408e4


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: mini-ctuning_cpp_tflite-cpu-tflite_vmaster-default_config

Model Precision: fp32

### Accuracy Results 
`acc`: `47.808`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `536482.0`

### Power Results 
`Power consumed`: `16.516 milliJoules`, `Power efficiency`: `60546.887 samples per Joule`
*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-53-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=828415392b4bd0ccaea54a041990ab875b341463


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: b8fc37015501-ctuning_cpp_tflite-cpu-tflite_vmaster-default_config

Model Precision: fp32

### Accuracy Results 
`acc`: `66.75`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `1612206.0`

*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-1021-aws-aarch64-with-glibc2.39
* CPU version: aarch64
* Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=2d6fc4b3487b6dc259092ad9808766e6757068f1


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: aws_m6g.large-ctuning_cpp_tflite-cpu-tflite_vmaster-default_config

Model Precision: fp32

### Accuracy Results 
`acc`: `66.604`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `11368971.0`

*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: macOS-15.3.1-arm64-arm-64bit
* CPU version: arm
* Python version: 3.9.6 (default, Nov 11 2024, 03:15:38) 
[Clang 16.0.0 (clang-1600.0.26.6)]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=997219a6b08ca993a732f7f262f39039868f81a7


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: MacBookPro.ultrahub-reference-cpu-deepsparse_v1.8.0-default_config

Model Precision: fp32

### Accuracy Results 
`F1`: `88.42042`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `28.6774`

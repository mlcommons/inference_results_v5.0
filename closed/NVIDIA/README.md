# MLPerf Inference v5.0 NVIDIA-Optimized Implementations
This is a repository of NVIDIA-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
This README is a quickstart tutorial on how to use our code as a public / external user.

---

### MLPerf Inference Policies and Terminology

This is a new-user guide to learn how to use NVIDIA's MLPerf Inference submission repo. **To get started with MLPerf Inference, first familiarize yourself with the [MLPerf Inference Policies, Rules, and Terminology](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)**. This is a document from the MLCommons committee that runs the MLPerf benchmarks, and the rest of all MLPerf Inference guides will assume that you have read and familiarized yourself with its contents. The most important sections of the document to know are:

- [Key terms and definitions](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#11-definitions-read-this-section-carefully)
- [Scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#3-scenarios)
- [Benchmarks and constraints for the Closed Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#411-constraints-for-the-closed-division)
- [LoadGen Operation](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#51-loadgen-operation)



### Quick Start on computelab

Rrequest gpu nodes on computelab

- [Machines on computelab](https://confluence.nvidia.com/display/GCA/MLPerf-Inference+v5.0+Machines)

`export MLPERF_SCRATCH_PATH=/path/to/scratch/space`: set mlperf scratch space

`make prebuild`: builds and launch the container.

`make build`: builds plugins and binaries.

`make generate_engines RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>`: generates engines.

`make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO>`: runs the harness to get perf results.

`make run_harness RUN_ARGS="--benchmarks=<BENCHMARK> --scenarios=<SCENARIO> --test_mode=AccuracyOnly`: runs the harness to get accuracy results.

Add --config_ver=high_accuracy to run with high accuracy target.

### NVIDIA's Submission

NVIDIA submits with multiple systems, each of which are in either the datacenter category, edge category, or both. In general, multi-GPU systems are submitted in datacenter, and single-GPU systems are submitted in edge.

Our submission implements several inference harnesses stored under closed/NVIDIA/code/harness:

- What we refer to as "custom harnesses": lightweight, barebones, C++/Python harnesses
    - 3D-UNET KiTS-19 harness
    - BERT harness
    - LWIS (Light-Weight Inference Server) - For ResNet50, RetinaNet
    - DLRMv2 harness
    - LLM harness (Python) - For Mixtral-8x7B, GPTJ, Llama2-70B and Llama3.1-405B
    - LLM harness (Cpp, deprecated) - For Mixtral-8x7B, GPTJ, Llama2-70B.
    - SDXL harness

Benchmarks are stored in `closed/NVIDIA/code`. Each benchmark, as per MLPerf Inference requirements, contains a `README.md` detailing instructions and documentation for that benchmark. **However**, as a rule of thumb, **follow this guide first** from start to finish before moving on to benchmark-specific `README`s, as this guide has many wrapper commands to automate the same steps across multiple benchmarks at the same time.

### Use a non-root user

**We highly recommend to run MLPerf as a sudo user (i.e. a user in the sudo group), and avoid using sudo command in the container. Some functionality might be broken without sudo privileges.**

If you're already a non-root user, simply don't use sudo for any command that is not a package install or a command that specifically has 'sudo' contained in it. Otherwise, create a new user. It is advisable to make this new user a sudoer, but as said before, do not invoke sudo unless necessary.

Make sure that your user is in docker group already. If you get permission issue when running docker commands, please add the user to docker group with `sudo usermod -a -G docker $USER`.

### Software Dependencies

### Datacenter systems

Our submission uses Docker to set up the environment. Requirements are:

- [Docker CE](https://docs.docker.com/engine/install/)
    - If you have issues with running Docker without sudo, follow this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for your new non-root user. Namely, add your new user to the Docker usergroup, and remove ~/.docker or chown it to your new user.
    - Install Docker buildx plugin: `apt-get install docker-buildx-plugin`
    - You may also have to restart the docker daemon for the changes to take effect:

```
$ sudo systemctl restart docker
```

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
    - libnvidia-container >= 1.4.0
- NVIDIA Driver Version 550.xx or greater

    - For v5.0 submission, we recommend driver version 550.xx or greater

### NUMA Configuration of the system

Correct NUMA configuration can help system performance by minimizing the distance between processors/accelerators and memory. Different vendors take different approaches on NUMA configurability of the underlying hardware configuration. In general, it is recommended to follow your system vendor's optimization guide.

It is recommended to:

- Split the system into NUMA nodes such that each NUMA node has the minimal possible latency for communication between all the processors/accelerators and memory subsystems. For example, AMD's EPYC architecture provides one Zeppelin to be one NUMA node. Configuring a NUMA node to have more than one Zeppelin would create a less efficient NUMA configuration.
- Maximize the memory bandwidth and capacity within the NUMA node. This is done by correctly populating the DIMMs on as many channels as possible in each node.
- Maximize the I/O bandwidth and capacity within the NUMA node. You will have to check PCIe lane availability to each NUMA node.
- Have symmetric configuration for both inter- and intra-node configurations.

Instructions to configure NUMA are different depending on CPU vendor:

- **AMD CPU**: NUMA configuration is done through NPC settings from BIOS. Please refer to: [https://developer.amd.com/wp-content/resources/56827-1-0.pdf](https://developer.amd.com/wp-content/resources/56827-1-0.pdf)
- **Intel CPU**: NUMA configuration is done through NUMA/UMA settings from BIOS. Please refer to: [https://software.intel.com/content/www/us/en/develop/articles/optimizing-applications-for-numa.html](https://software.intel.com/content/www/us/en/develop/articles/optimizing-applications-for-numa.html)
- **ARM CPU**: NUMA configuration is not yet supported as most of the systems currently built with ARM CPUs are single socket and single package.

### Setting up the Scratch Spaces

NVIDIA's MLPerf Inference submission stores the models, datasets, and preprocessed datasets in a central location we refer to as a "Scratch Space".

Because of the large amount of data that needs to be stored in the scratch space, we recommend that the scratch be at least **10 TB**. This size is recommended if you wish to obtain every dataset in order to run each benchmark and have extra room to store logs, engines, etc. If you do not need to run every single benchmark, it is possible to use a smaller scratch space.


**Note that once the scratch space is setup and all the data, models, and preprocessed datasets are set up, you do not have to re-run this step.** You will only need to revisit this step if:

- You accidentally corrupted or deleted your scratch space
- You need to redo the steps for a benchmark you previously did not need to set up
- You, NVIDIA, or MLCommons has decided that something in the preprocessing step needed to be altered

Once you have obtained a scratch space, set the `MLPERF_SCRATCH_PATH` environment variable. This is how our code tracks where the data is stored. By default, if this environment variable is not set, we assume the scratch space is located at `/home/mlperf_inference_storage`. Because of this, it is highly recommended to mount your scratch space at this location.


**If you export MLPERF_SCRATCH_PATH, scratch space will mount automatically when you launch container.**

```
$ export MLPERF_SCRATCH_PATH=/path/to/scratch/space
```
This `MLPERF_SCRATCH_PATH` will also be mounted inside the docker container at the same path (i.e. if your scratch space is located at `/mnt/some_ssd`, it will be mounted in the container at `/mnt/some_ssd` as well.)

Then create empty directories in your scratch space to house the data:

```
$ mkdir $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data
```
After you have done so, you will need to download the models and datasets, and run the preprocessing scripts on the datasets. **If you are submitting MLPerf Inference with a low-power machine, such as a mobile platform, it is recommended to do these steps on a desktop or server environment with better CPU and memory capacity.**

Enter the container by entering the `closed/NVIDIA` directory and running:

```
$ make prebuild # Builds and launches a docker container
```
Then inside the container, you will need to do the following:

```
$ echo $MLPERF_SCRATCH_PATH  # Make sure that the container has the MLPERF_SCRATCH_PATH set correctly
$ ls -al $MLPERF_SCRATCH_PATH  # Make sure that the container mounted the scratch space correctly
$ make clean  # Make sure that the build/ directory isn't dirty
$ make link_dirs  # Link the build/ directory to the scratch space
$ ls -al build/  # You should see output like the following:
total 8
drwxrwxr-x  2 user group 4096 Jun 24 18:49 .
drwxrwxr-x 15 user group 4096 Jun 24 18:49 ..
lrwxrwxrwx  1 user group   35 Jun 24 18:49 data -> $MLPERF_SCRATCH_PATH/data
lrwxrwxrwx  1 user group   37 Jun 24 18:49 models -> $MLPERF_SCRATCH_PATH/models
lrwxrwxrwx  1 user group   48 Jun 24 18:49 preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
```
Once you have verified that the `build/data`, `build/models/`, and `build/preprocessed_data` point to the correct directories in your scratch space, you can continue.

### Prepping our repo for your machine

We formally support and fully test the configuration files for the following systems:

Datacenter systems:

- H100-SXM-80GBx8 (NVIDIA DGX H100)
- H200-SXM-144GBx8 (NVIDIA DGX H200)
- GH200 Grace Hopper Superchip
- B200-SXM-180GBx8 (NVIDIA DGX B200)

Edge Systems:

- N/A

**If your system is not listed above, nor listed in the `code/common/systems/system_list.py`, you must add your system to our 'KnownSystem' list.**

From v2.0 onwards, this step is now automated by a new script located in `scripts/custom_systems/add_custom_system.py`. See the 'Adding a New or Custom System' section further down.

## Running your first benchmark

**First, enter closed/NVIDIA**. From now on, all of the commands detailed in this guide should be executed from this directory. This directory contains our submission code for the [MLPerf Inference Closed Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#61-closed-division). NVIDIA may also submit under the [MLPerf Inference Open Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#63-open-division) as well, and many of the commands in the Open Division are the same, but there are many nuances specific to the "open" variants of certain benchmarks.

***IMPORTANT**:* **Do not run any commands as root (Do not use sudo) unless written in NVIDIA-provided scripts.** Running under root messes up a lot of permissions, and has caused many headaches in the past. If for some reason you missed the part in the beginning of the guide that warned to not use root, you may run into one of the following problems:

- Your non-root account cannot use Docker.
    - See the 'Use a non-root user' section at the beginning of the guide for instructions on how to fix this.
- You cloned the repo as root, now you have a bunch of file permission errors where you cannot write to some directories.
    - It is highly recommended to chown the entire repo to the new non-root user, or better yet to re-clone the repo with the new user.
    - You will likely also need to re-run the 'git config' and 'Docker login' steps in the 'Cloning the Repo' Section, as those are user-specific commands, and would only have affected 'root'.
- Make sure that your new user has at least read-access to the scratch spaces. If the scratch space was set up incorrectly, only 'root' will be able to read the scratch spaces. If the scratch spaces are network-based filesystems, check /etc/fstab for the settings as well.

### Launching the docker environment

If you are on a desktop, server system or Jetson embedded system, you will need to launch the Docker container first:

```
$ make prebuild
```
***Important notes:***

- **If you are an NVIDIA partner engineer with access to NVIDIA's partner private NGC registry, please use `$ PARTNER_DROP=1 make prebuild`**
- The docker container does not copy the files, and instead **mounts** the working directory (closed/NVIDIA) under /work in the container. This means you can edit files outside the container, and the changes will be reflected inside as well.
- In addition to mounting the working directory, the scratch spaces are also mounted into the container. Likewise, this means if you add files to the scratch spaces outside the container, it will be reflected inside the container and vice versa.
- If you want to mount additional directories/spaces in the container, use `$ DOCKER_ARGS="-v <from>:<to> -v <from>:<to>" make prebuild `
- If you want to expose only a certain number of GPUs in the container, use `$ NVIDIA_VISIBLE_DEVICES=0,2,4... make prebuild`

### Adding a New or Custom System

To add a new system, from inside the docker container, run:

```
$ python3 -m scripts.custom_systems.add_custom_system
```
This script will first show you the information of the detected system, like so:

```
============= DETECTED SYSTEM ==============

SystemConfiguration:
    System ID (Optional Alias): None
    CPUConfiguration:
        2x CPU (CPUArchitecture.x86_64): AMD EPYC 7742 64-Core Processor
            64 Cores, 2 Threads/Core
    MemoryConfiguration: 990.59 GB (Matching Tolerance: 0.05)
    AcceleratorConfiguration:
        2x GPU (0x20B710DE): NVIDIA A30
            AcceleratorType: Discrete
            SM Compute Capability: 80
            Memory Capacity: 24.00 GiB
            Max Power Limit: 165.0 W
    NUMA Config String: &&&&&&1:96-111,224-239&0:112-127,240-255

============================================
```
If the detected system is already in known, the script will print a warning like so:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: The system is already a known submission system (KnownSystem.A100_SXM_80GBx1).
You can either quit this script (Ctrl+C) or continue anyway.
Continuing will perform the actions described above, and the current system description will be replaced.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
In this case, you should quit the script by either entering `n` at the prompt, or by pressing Ctrl+C.

Otherwise, the script will ask you to enter a system ID to use for this new system. This system ID will be the name that appears in your **results, measurements, and systems directories in your submission** for the current system.

After entering a system ID, the script will generate (or append to, if already existing) a file at `code/common/systems/custom_list.py`. This is an example snippet of the generated line:

```
# Do not manually edit any lines below this. All such lines are generated via scripts/add_custom_system.py

###############################
### START OF CUSTOM SYSTEMS ###
###############################

custom_systems['A30x4_Custom'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7742 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=64, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=990.594852, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 4}), numa_conf=None, system_id="A30x4_Custom")

###############################
#### END OF CUSTOM SYSTEMS ####
###############################
```
**If later, you wish to remove a system**, simply edit this file and delete the line it is defined in, as well as all associated benchmark configs. **If you re-use a System ID, it will use the most recent definition** as the runtime value. This way, you can actually redefine existing NVIDIA submission systems to match your systems if you want to use that particular system ID.

The script will then ask you if you want to generate stubs for the Benchmark Configuration files, located in `configs/`. If this is your first time running NVIDIA's MLPerf Inference v3.0 code for this system, enter `y` at the prompt. This will generate stubs for every single benchmark, located at `configs/[benchmark]/[scenario]/custom.py`. An example stub is below:

```
# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    input_dtype: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    offline_expected_qps: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    soft_drop: float = 0.0
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    workspace_size: int = 0
```
These stubs will show all of the fields that can pertain to the benchmark, divided into required and optional sections. Most of the time, you can ignore this, and simply copy over the fields from an existing submission. In this example, my custom system is an A30x4 machine, so I can copy over the A30x1 BERT offline configuration like so (keeping the `system = KnownSystem.A30x4_Custom`):

```
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = {'bert': 1024}
    offline_expected_qps = 1972 * 4  # Here, I add *4 since I copied the original QPS from the A30x1 config.
    workspace_size = 7516192768
```
Alternatively, **if your system uses a GPU that is already supported by NVIDIA's MLPerf Inference Submission**, you can simply extend one of NVIDIA's configs and override some values. In this case, our A30x4 config can extend A30x1 instead of `OfflineGPUBaseConfig`, and just redefine the `system` and `offline_expected_qps` fields:

```
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM(A30x1):
    system = KnownSystem.A30x4_Custom
    offline_expected_qps = 1972 * 4  # Here, I add *4 since I copied the original QPS from the A30x1 config.
```
You can see the GPUs that NVIDIA supports by looking in the `KnownGPU` Enum located in `code/common/systems/known_hardware.py`.

### Building the binaries

```
$ make build
```
This command does several things:

1. Sets up symbolic links to the models, datasets, and preprocessed datasets in the MLPerf Inference scratch space in build/
2. Pulls the specified hashes for the subrepositories in our repo:
    1. MLCommons Inference Repo (Official repository for MLPerf Inference tools, libraries, and references)
3. Builds all necessary binaries for the specific detected system

**Note**: This command does not need to be run every time you enter the container, as build/ is stored in a mounted directory from the host machine. It does, however, need to be re-run if:

- Any changes are made to harness code
- Repository hashes are updated for the subrepositories we use
- You are re-using the repo on a system with a different CPU architecture

### Running the actual benchmark

Our repo has one main command to run any of our benchmarks:

```
$ make run RUN_ARGS="..."
```
This command is actually shorthand for a 2-step process of building, then running TensorRT engines:

```
$ make generate_engines RUN_ARGS="..."
$ make run_harness RUN_ARGS="..."
```
By default, if RUN_ARGS is not specified, this will run every system-applicable benchmark-scenario pair under submission settings. This means it will run 6 benchmarks * 2 scenarios * up to 4 variations = up to 48 workloads, each with a minimum runtime of 10 minutes.

This is not ideal, as that can take a while, so RUN_ARGS supports a --benchmarks and --scenarios flag to control what benchmarks and scenarios are run. These flags both take comma-separated lists of names of benchmarks and scenarios, and will run the cartesian product of these 2 lists.

Valid benchmarks are:

- resnet50
- retinanet
- bert (edge only)
- 3d-unet (no server or multistream)
- dlrm-v2 (datacenter system only)
- gptj
- llama2-70b (datacenter system only)
- llama2-70b-interactive (datacenter system only, note this shares the same code/harness/data/model as llama2-70b)
- sdxl
- mixtral-8x7b (datacenter system only)
- llama3.1-405b (datacenter system only)

Valid scenarios are:

- offline
- server (datacenter system only)
- singlestream (edge system only)
- multistream (edge system only)

**Example**:

To run ResNet50, and BERT under the Offline and Server scenarios:

```
$ make run RUN_ARGS="--benchmarks=resnet50,bert --scenarios=offline,server"
```
**If you run into issues, invalid results, or would like to improve your performance,** **read** `documentation/performance_tuning_guide.md`.

### Multinode runs

From within the container, installing the triton software:

```
$ make clone_triton && make build_triton
```

Generating the triton config files(after generating the engines):

```
$ make generate_triton_config RUN_ARGS="--benchmarks=llama2-70b \ #or other benchmarks
 --scenarios=Offline \
--harness_type=triton \
--accuracy_target=0.999 \ # or 0.99
--engine_dir=/path/to/engines \”
```

After modifying the `start_triton.sh` based on your specific node config, start the triton engines on each node:

```
$ /work/start_triton.sh
```

Keeping the engines running, enter one node's(considered the master node hereon) container from a different shell:

```
$ docker exec -it "image-name" bash
```

Start the accuracy/throughput runs through seperate triton-client frontends for each node:

```
$ make run_harness RUN_ARGS="\
  --benchmarks=llama2-70b \ # or the other benchmarks
  --scenarios=Offline \
  --harness_type=triton \
  --inference_server=triton \
  --accuracy_target=0.999 \ # or 0.99
  --triton_skip_server_spawn \
  --triton_grpc_ports='ip_of_master_node:8001|ip_of_worker_node_1:8001|ip_of_worker_node_2:8001...'"
```


### How do I run the accuracy checks?

You can run the harness for accuracy checks using the `--test_mode=AccuracyOnly` flag:

```
$ make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --test_mode=AccuracyOnly"
```
### Do I have to run with a minimum runtime of 10 minutes? That is a really long time.

Yes and no. Effective v1.0 of MLPerf Inference, it is **required** for the SUT (System Under Test) to run the workload for a minimum of 10 minutes to be considered a valid run for submission. This duration was chosen to allow ample time for the system to reach thermal equilibrium, and to reduce possible variance caused by the load generation.

However, for development and quick sanity checking we provide an optional **--test_run** flag that can be added to RUN_ARGS that will reduce the minimum runtime of the workload from 10 minutes to 1 minute (which was the minimum duration before v1.0).

Ex. To run RetinaNet Offline for a minimum of 1 minute instead of 10:

```
$ make run RUN_ARGS="--benchmarks=retinanet --scenarios=offline --test_run"
```
### How do I view the logs of my previous runs?

Logs are saved to `build/logs/[timestamp]/[system ID]/...` every time `make run_harness` is called.

### Make run is wasting time re-building engines every time for the same workload. Is this required?

Nope! You only need to build the engine once. You can either call `make generate_engines`, or `make run` first for your specified workload. Afterwards, to run the engine, just use `make run_harness` instead of `make run`.

**Re-building engines is only required if**

- You ran `make clean` or deleted the engine
- It is a new workload that hasn't had an engine built yet
- You changed some builder settings in the code
- You updated the TensorRT or TensorRT LLM version (i.e. a new partner drop)
- You updated the benchmark configuration with a new batch size or engine-build-related setting

### Building and running engines for the "High Accuracy Target"

In MLPerf Inference, there are a few benchmarks that have a second "mode" that requires the benchmark to pass with at least 99.9% of FP16/FP32 accuracy. In our code, we refer to the normal accuracy target of 99% of FP16/FP32 as 'default' or 'low accuracy' mode, and we refer to the 99.9% of FP16/FP32 target as 'high accuracy' mode.

The following benchmarks have '99.9% FP16/FP32' variants:

- DLRM-v2
- BERT
- 3D UNET
- GPTJ
- Llama2-70B
- Llama2-70B-interactive

To run the benchmarks under the higher accuracy target, specify `--config_ver="high_accuracy"` as part of `RUN_ARGS`:

```
$ make run RUN_ARGS="--benchmarks=dlrm-v2,3d-unet,bert --scenarios=offline --test_run --config_ver=high_accuracy"
```
Note that you will also have to run the generate_engines step with this config_ver, as it is possible the high accuracy target requires different engine parameters (i.e. requiring FP16 precision instead of INT8).

If you want to run the accuracy tests as well, you can use the `--test_mode=AccuracyOnly` flag as normal.

### How to collect power measurements while running the harness

Starting in MLPerf Inference v1.0, the 'power measurement' category introduced a new metric for judging system performance: Perf per watt. In this setting, rather than comparing systems' peak throughput, it compares the ratio between the peak throughput and the power usage of the machine during inference.

NVIDIA's power category submission is called 'MaxQ'. To run the harness with power measurements, follow the steps below:

1. Set the machine to the desired power mode. The Makefile in closed/NVIDIA contains a target called `power_set_maxq_state` that describes the power settings we use for our MaxQ submissions. Run this make target before proceeding.
2. Set up a Linux power director machine with the following requirements.
    1. PTDaemon `ptd-linux-x86` is installed in `/usr/bin`. The PTDaemon executable file is located in a private repo: [https://github.com/mlcommons/power](https://github.com/mlcommons/power) and submitters must join the Power WG and sign the PTD EULA license agreement to get it.
    2. The MLCommons [power-dev repo](https://github.com/mlcommons/power-dev) is cloned in `~/mlperf-power/power-dev` and is on the correct branch for the MLPerf Inference version (i.e. `r1.0` for MLPerf Inference v1.0)
    3. You have created a directory at `~/power_logs`
    4. There exists an administrator user `lab` with password `labuser-mlpinf`. If your administrator account has different login information, set `POWER_SERVER_USERNAME` and `POWER_SERVER_PASSWORD` in `closed/NVIDIA/Makefile.power` to the correct credentials.
    5. OpenSSH server is installed and enabled on the machine, listening on port 22.
    6. Set the IP of the power system in `power/power_server_ips.py`
3. Set the power meter configuration in `power/server-$HOSTNAME.cfg`. Refer to NVIDIA sample configuration (`server-template-linux.cfg`) files as examples, and note that you need to bump up the listen port and network port so different systems don't collide with each other.
4. Instead of `make run_harness`, use `make run_harness_power` for PerformanceOnly mode. All other commands will work, but if you run `make run_harness` instead of `run_harness_power`, it will run the harness without power measurements. With this make target, LoadGen logs will be located in `build/power_logs` instead of `build/logs`. The commands for AccuracyOnly mode and the audit tests remain unchanged.

    1. When `make run_harness_power` is called, the script runs the harness twice: the first run is called the "ranging run", which is used to gather the maximum voltage and current that the system consumes so as to configure the power meter correctly. Then, the second run is called the "testing run" which actually collects the power readings.
    2. The default run will launch Linux PTD and server. Add `USE_WIN_PTD=1` to use windows path.
5. In NVIDIA's submission, we use the Yokogawa WT333E meter in either single channel or multi-channel mode. Please refer to `power/server-$HOSTNAME.cfg` for what mode is used on which machine.
6. To update the logs in the results/ and the compliance/ directories, use the same commands like the non-power submission by running `make update_results` and `make update_compliance`, respectively. The logs in `build/power_logs` will be automatically copied to `results/` directory if the logs are valid.


### Running INT8 calibration

**Note this is not needed for external users**

For legacy benchmarks using implicit quantization (e.g. RetinaNet, 3d-unet etc.), the calibration caches generated from the default calibration sets (set by MLPerf Inference committee) are already provided in each benchmark directory. If you would like to regenerate the calibration cache for a specific benchmark, run:

```
$ make calibrate RUN_ARGS="--benchmarks=[benchmark]"
```
See documentation/calibration.md for an explanation on how calibration is used for NVIDIA's submission.

### Update the results directory for submission

Refer to documentation/submission_guide.md.

### Run compliance tests and update the compliance test logs

Refer to documentation/submission_guide.md.

### Preparing for submission

MLPerf Inference policies as of v1.0 include an option to allow submitters to submit an encrypted tarball of their submission repository, and share a SHA1 of the encrypted tarball and the decryption password with the MLPerf Inference results chair. This option gives submitters a more secure, private submission process. NVIDIA and **all NVIDIA partners** must use this new submission process to ensure fairness among submitters.

For instructions on how to encrypt your submission, see the 'Encrypting your project for submission' section of documentation/submission_guide.md.

**IMPORTANT**: In v2.0, the MLPerf Inference committee is working to put together a web-based submission page so that you can submit your results from the website. This webpage will have an option to use an **encrypted** submission. **ALL NVIDIA Submission partners** are expected to use this encrypted submission to **avoid leaking results** to competitors. As of **1/24/2022**, this webpage has not yet been finalized, so the instructions for actually submitting your results tarball are **outdated and incorrect**. When the page and the URL have been finalized, NVIDIA will notify partners of the correct submission instructions.

### Instructions for Auditors

Please refer to the README.md in each benchmark directory for auditing instructions.

### Download the datasets

** Internal MLPerf dataset only need to be setup once from admin. **

Each benchmark contains a `README.md` (located at `closed/NVIDIA/code/[benchmark name]/tensorrt/README.md`) that explains how to download and set up the dataset and model files for that benchmark manually. **We recommend that you at least read the README.md files for benchmarks that you plan on running or submitting.** However, you do not need to actually follow the instructions in these READMEs as instructions to automate the same steps across multiple benchmarks are detailed below.

**Note that you do not need to download the datasets or models for benchmarks that you will not be running.**

While we have some commands and scripts to automate this process, **some benchmarks use datasets that are not publicly available**, and are gated by license agreements or signup forms. For these benchmarks, **you must retrieve the datasets manually**:

- `ResNet50`: Download the [ImageNet 2012 Validation Set](https://image-net.org/challenges/LSVRC/2012/) and unzip the files to `$MLPERF_SCRATCH_PATH/data/imagenet/`.
- `DLRM-v2`: Download the [Criteo Terabyte dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) and unzip the files to `$MLPERF_SCRATCH_PATH/data/criteo/`.
- `3D UNET`: Clone the [KiTS19 GitHub repository](https://github.com/neheller/kits19) into `$MLPERF_SCRATCH_PATH/data/KiTS19/` as described in [KiTS 2019 Kidney Tumor Segmentation set](https://kits19.grand-challenge.org/).
- `Llama2-70B`: Please refer to the `code/llama2-70b/tensorrt/README.md`
- `Llama3.1-405B`: Please refer to the `code/llama3_1-405b/tensorrt/README.md`

After you have downloaded all the datasets above that you need, the rest can be automated by using:

```
$ make download_data # Downloads all datasets and saves to $MLPERF_SCRATCH_PATH/data
```
If you only want to download the datasets for specific models, you can specify use the `BENCHMARKS` environment variable:

```
# Specify BENCHMARKS="space separated list of benchmarks"
# The below is the default value of BENCHMARKS, and is equivalent to not specifying BENCHMARKS:
$ make download_data BENCHMARKS="resnet50 retinanet bert 3d-unet"

# Remove the benchmarks you don't want. If you only want to run the retinanet and bert, do:
$ make download_data BENCHMARKS="retinanet bert"
```
Note that if the dataset for a benchmark already exists, the script will print out a message confirming that the directory structure is as expected.

If you specified a benchmark that does not have a public dataset **and did not manually download and extract it**, you will see a message like:

```
!!!! Dataset cannot be downloaded directly !!!
Please visit [some URL] to download the dataset and unzip to [path].
Directory structure:
    some/path/...
```
This is expected, and you should follow the instructions detailed to retrieve the dataset. **If you do not need to run that benchmark, you can ignore this error message.**

### Downloading the model files

Most of the model downloading is automated, with the following exception:
- `llama2-70B`: Please refer to the `code/llama2-70b/tensorrt/README.md`
- `mixtral-8x7b`: Please refer to the `code/mixtral-8x7b/tensorrt/README.md`
- `Llama3.1-405B`: Please refer to the `code/llama3_1-405b/tensorrt/README.md`

For the remaining of the models, we provide the following command to download the models via command line. Note that you can use the same optional `BENCHMARK` argument as in the 'Download the datasets' section:

```
$ make download_model BENCHMARKS="resnet50 retinanet bert 3d-unet gptj stable-diffusion-xl"
```
Just like when you downloaded the datasets, remove any of the benchmarks you do not need from the list of benchmarks.

**Before proceeding, double check that you have downloaded both the dataset AND model for any benchmark you are planning on running.**

### Preprocessing the datasets for inference

NVIDIA's submission preprocesses the datasets to prepare them for evaluation. These are operations like the following:

- Converting the data to INT8 or FP16 byte formats
- Restructuring the data channels (i.e. converting images from NHWC to NCHW)
- Saving the data as a different filetype, usually serialized NumPy arrays

Just like the prior 2 steps, there is a command to automate this process that also takes the same BENCHMARKS argument:

```
$ make preprocess_data BENCHMARKS="resnet50 retinanet bert 3d-unet gptj llama2-70b stable-diffusion-xl mixtral-8x7b llama3_1-405b"
```
**As a warning, this step can be very time consuming and resource intensive depending on the benchmark.**

**Note**: the above steps (*Download the datasets, Downloading the model files, Preprocessing the datasets for inference*) are **not** guaranteed to work on Jetson-based system (e.g. Orin). It is suggested to run the steps on other cuda-enabled devices, and copy over the $(MLPERF_SCRATCH_PATH)/ directory if needed. If any target fails, please try to run it inside the container following *Launching the environment* section below.


### Further reading

More specific documentation and for debugging:

- documentation/performance_tuning_guide.md - Documentation related to tuning and benchmarks via configuration changes
- documentation/commands.md - Documentation on commonly used Make targets and RUN_ARGS options
- documentation/FAQ.md - An FAQ on common errors or issues that have popped up in the past
- documentation/submission_guide.md - Documentation on officially submitting our repo to MLPerf Inference
- documentation/calibration.md - Documentation on how we use calibration and quantization for MLPerf Inference


# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(SingleStreamGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    map_path: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    cache_file: str = ''
    complete_threads: int = 0
    engine_dir: str = ''
    single_stream_expected_latency_ns: int = 0
    single_stream_target_latency_percentile: float = 0.0
    slice_overlap_patch_kernel_cg_impl: bool = False
    unet3d_sw_gaussian_patch_path: str = ''
    use_batcher_thread_per_device: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_same_context: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy(C245M8_H100NVL_94GBX2):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_Triton(C245M8_H100NVL_94GBX2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    map_path: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    cache_file: str = ''
    complete_threads: int = 0
    engine_dir: str = ''
    gather_kernel_buffer_threshold: int = 0
    max_queue_delay_usec: int = 0
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    output_pinned_memory: bool = False
    single_stream_expected_latency_ns: int = 0
    single_stream_target_latency_percentile: float = 0.0
    slice_overlap_patch_kernel_cg_impl: bool = False
    triton_grpc_ports: str = ''
    triton_num_clients_per_frontend: int = 0
    triton_num_frontends_per_model: int = 0
    triton_num_servers: int = 0
    triton_skip_server_spawn: bool = False
    triton_verbose_frontend: bool = False
    unet3d_sw_gaussian_patch_path: str = ''
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_same_context: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy_Triton(C245M8_H100NVL_94GBX2_HighAccuracy):
    use_triton = True



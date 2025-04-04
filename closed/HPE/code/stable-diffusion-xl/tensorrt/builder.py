# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from os import PathLike
from pathlib import Path

import os
import onnx
import tempfile
import tensorrt as trt

import polygraphy.logger
from typing import Dict, List, Optional
from importlib import import_module

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    modify_network_outputs,
    save_engine
)

from nvmitten.constants import Precision
from nvmitten.nvidia.builder import (TRTBuilder,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from nvmitten.utils import dict_get, logging

from code.common.mitten_compat import ArgDiscarder
from code.common.systems.system_list import DETECTED_SYSTEM

# dash in stable-diffusion-xl breaks traditional way of module import
AbstractModel = import_module("code.stable-diffusion-xl.tensorrt.network").AbstractModel
CLIP = import_module("code.stable-diffusion-xl.tensorrt.network").CLIP
CLIPWithProj = import_module("code.stable-diffusion-xl.tensorrt.network").CLIPWithProj
UNetXL = import_module("code.stable-diffusion-xl.tensorrt.network").UNetXL
VAE = import_module("code.stable-diffusion-xl.tensorrt.network").VAE
SDXLGraphSurgeon = import_module("code.stable-diffusion-xl.tensorrt.sdxl_graphsurgeon").SDXLGraphSurgeon
SDXLComponent = import_module("code.stable-diffusion-xl.tensorrt.constants").SDXLComponent
polygraphy.logger.G_LOGGER.module_severity = polygraphy.logger.G_LOGGER.ERROR


class SDXLBaseBuilder(TRTBuilder,
                      MLPerfInferenceEngine):
    """Base SDXL builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int = 1,
                 model: AbstractModel,
                 model_path: str,
                 workspace_size: int = 80 << 30,
                 num_profiles: int = 1,  # Unused. Forcibly overridden to 1.
                 device_type: str = "gpu",
                 strongly_typed: bool = False,
                 use_native_instance_norm: bool = False,
                 **kwargs):
        # TODO: yihengz Force num_profiles to 1 for SDXL, not sure if multiple execution context can help heavy benchmarks
        super().__init__(*args, num_profiles=1, workspace_size=workspace_size, **kwargs)

        self.model = model
        self.model_path = model_path
        # engine precision is determined by the model
        if self.model.precision == 'fp32':
            self.precision = Precision.FP32
        elif self.model.precision == 'fp16':
            self.precision = Precision.FP16
        elif self.model.precision == 'int8':
            self.precision = Precision.INT8
        elif self.model.precision == 'fp8':
            self.precision = Precision.FP8
        else:
            raise ValueError("Unsupported model precision")
        self.batch_size = batch_size
        self.device_type = device_type
        self.strongly_typed = strongly_typed
        self.use_native_instance_norm = use_native_instance_norm
        self.skip_gs = False

    def create_network(self, builder: trt.Builder = None) -> trt.INetworkDefinition:
        if self.strongly_typed:
            network = super().create_network(builder=builder, flags=1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        else:
            network = super().create_network(builder=builder)

        parser = trt.OnnxParser(network, self.logger)

        # set instance norm flag for better perf of SDXL
        if self.use_native_instance_norm:
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

        if self.skip_gs:
            success = parser.parse_from_file(str(self.model_path))
            if not success:
                err_desc = parser.get_error(0).desc()
                raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")
        else:
            add_hidden_states = isinstance(self.model, CLIP) or isinstance(self.model, CLIPWithProj)
            sdxl_gs = SDXLGraphSurgeon(self.model_path,
                                       self.precision,
                                       self.device_type,
                                       self.model.name,
                                       add_hidden_states=add_hidden_states)

            model = sdxl_gs.create_onnx_model()

            if model.ByteSize() >= SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
                # onnx._serialize cannot take input proto >= 2 BG
                # We need to save proto larger than 2GB into separate files and parse from files
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    tmp_path.mkdir(exist_ok=True)
                    onnx_tmp_path = tmp_path / "tmp_model.onnx"
                    onnx.save_model(model,
                                    str(onnx_tmp_path),
                                    save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    convert_attribute=False)
                    success = parser.parse_from_file(str(onnx_tmp_path))
                    if not success:
                        err_desc = parser.get_error(0).desc()
                        raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")
            else:
                # Parse from ONNX file
                success = parser.parse(model.SerializeToString())
                if not success:
                    err_desc = parser.get_error(0).desc()
                    raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")

        if not (self.model.name == "vae" and int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100):
            logging.info(f"Updating network outputs to {self.model.get_output_names()}")
            _, network, _ = modify_network_outputs((self.builder, network, parser), self.model.get_output_names())

        if not self.strongly_typed:
            self.apply_network_io_types(network)
        return network

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set input dtype
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if self.precision == Precision.FP32:
                input_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                input_tensor.dtype = trt.float16

        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                output_tensor.dtype = trt.float16

    # Overwrites mitten function with the same signature, `network` is unused
    def gpu_profiles(self,
                     network: trt.INetworkDefinition,
                     batch_size: int):
        profile = Profile()
        input_profile = self.model.get_input_profile(batch_size)
        for name, dims in input_profile.items():
            assert len(dims) == 3
            # for Blackwell make vae input shape be static shape for accuracy optimization.
            profile.add(name, min=dims[0], opt=dims[1], max=dims[2]) if not (self.model.name == "vae" and int(
                DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100) else profile.add(name, min=dims[1], opt=dims[1], max=dims[2])
        return [profile]

    def create_builder_config(self,
                              workspace_size: Optional[int] = None,
                              profiles: Optional[List[Profile]] = None,
                              precision: Optional[Precision] = None,
                              **kwargs) -> trt.IBuilderConfig:
        # TODO: yihengz explore if builder_optimization_level = 5 can get better perf, disabling for making engine build time too long
        if precision is None:
            precision = self.precision
        if workspace_size is None:
            workspace_size = self.workspace_size

        if self.strongly_typed:
            builder_config = CreateConfig(
                profiles=profiles,
                tf32=True,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED if (self.verbose or self.verbose_nvtx) else trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_size}
            )
        else:
            builder_config = CreateConfig(
                profiles=profiles,
                int8=precision == Precision.INT8,
                fp16=precision == Precision.FP16 or precision == Precision.INT8,
                tf32=precision == Precision.FP32,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED if (self.verbose or self.verbose_nvtx) else trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_size}
            )

        return builder_config

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,  # created inside
                     save_to: PathLike):
        save_to = Path(save_to)
        if save_to.is_file():
            logging.warning(f"{save_to} already exists. This file will be overwritten")
        save_to.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Building TensorRT engine for {self.model_path}: {save_to}")

        engine = engine_from_network(
            (self.builder, network),
            config=builder_config,
        )

        if self.verbose:
            engine_inspector = engine.create_engine_inspector()
            layer_info = engine_inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
            logging.info("========= TensorRT Engine Layer Information =========")
            logging.info(layer_info)

            # [https://nvbugs/3965323] Need to delete the engine inspector to release the refcount
            del engine_inspector

        save_engine(engine, path=save_to)

# TODO yihengz check if we can remove max_batch_size


class SDXLCLIPBuilder(SDXLBaseBuilder,
                      ArgDiscarder):
    """SDXL CLIP builder class.
    """

    def __init__(self,
                 *args,
                 component_name: str,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):
        clip_path = model_path + "onnx_models/clip1/model.onnx"
        clip_precision = 'fp32' if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100 else 'fp16'
        super().__init__(*args,
                         model=CLIP(name=component_name, max_batch_size=batch_size, precision=clip_precision, device='cuda'),
                         model_path=clip_path,
                         batch_size=batch_size,
                         **kwargs)

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIP keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


class SDXLCLIPWithProjBuilder(SDXLBaseBuilder,
                              ArgDiscarder):
    """SDXL CLIPWithProj builder class.
    """

    def __init__(self,
                 *args,
                 component_name: str,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):
        clip_proj_path = model_path + "onnx_models/clip2/model.onnx"
        clip_precision = 'fp32' if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100 else 'fp16'

        super().__init__(*args,
                         model=CLIPWithProj(name=component_name, max_batch_size=batch_size, precision=clip_precision, device='cuda'),
                         model_path=clip_proj_path,
                         batch_size=batch_size,
                         **kwargs)

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIPWithProj keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


class SDXLUNetXLBuilder(SDXLBaseBuilder,
                        ArgDiscarder):
    """SDXL UNetXL builder class.
    """

    def __init__(self,
                 *args,
                 component_name: str,
                 batch_size: int,  # *2 for prompt + negative prompt
                 precision: str,
                 model_path: PathLike,
                 **kwargs):

        if precision == 'fp8':
            unetxl_path = model_path + "modelopt_models/unetxl.fp8/unet.onnx"
            strongly_typed = True
        elif precision == 'int8':
            unetxl_path = model_path + "modelopt_models/unetxl.int8/unet.onnx"
            strongly_typed = False
        elif precision == 'fp16':
            unetxl_path = model_path + "onnx_models/unetxl/model.onnx"
            strongly_typed = False
        else:
            raise ValueError("Unsupported UNetXL precision")

        super().__init__(*args,
                         model=UNetXL(name=component_name, max_batch_size=batch_size, precision=precision, device='cuda'),
                         model_path=unetxl_path,
                         batch_size=batch_size,
                         strongly_typed=strongly_typed,
                         **kwargs)

        # Orin unet int8 will run OOM in GS, so we skip GS
        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            self.skip_gs = True


class SDXLVAEBuilder(SDXLBaseBuilder,
                     ArgDiscarder):
    """SDXL VAE builder class.
    """

    def __init__(self,
                 *args,
                 component_name: str,
                 batch_size: int,
                 model_path: PathLike,
                 **kwargs):

        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            vae_precision = 'fp32'
            vae_path = model_path + "onnx_models/vae/model.onnx"
            strongly_typed = False
        elif int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            vae_precision = 'fp32'
            vae_path = model_path + "onnx_models/vae/model.onnx"
            strongly_typed = False
        else:
            vae_precision = 'int8'
            vae_path = model_path + "modelopt_models/vae.int8/vae.onnx"
            strongly_typed = True
        super().__init__(*args,
                         model=VAE(name=component_name, max_batch_size=batch_size, precision=vae_precision, device='cuda'),
                         model_path=vae_path,
                         batch_size=batch_size,
                         strongly_typed=strongly_typed,
                         use_native_instance_norm=True,
                         **kwargs)
        # for accuracy issue B200 do not go through gs
        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            self.skip_gs = True


class SDXLBuilderManager:
    def __init__(self, builder):
        self.builder = builder

    def __enter__(self):
        if "unet" == self.builder.model.name and int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            logging.info(f"Building {self.builder.model.name} with blacklist.")
            os.system('sudo nvidia-smi -lgc 1980')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if "unet" == self.builder.model.name and int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            os.system('sudo nvidia-smi -rgc')

        return False


class SDXLEngineBuilderOp(Operation, ArgDiscarder):
    COMPONENT_BUILDER_MAP = {
        SDXLComponent.CLIP1: SDXLCLIPBuilder,
        SDXLComponent.CLIP2: SDXLCLIPWithProjBuilder,
        SDXLComponent.UNETXL: SDXLUNetXLBuilder,
        SDXLComponent.VAE: SDXLVAEBuilder,
    }

    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 *args,
                 batch_size: Dict[SDXLComponent, int] = None,
                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",
                 model_path: str = "build/models/SDXL/",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_dir = dict_get(kwargs, "engine_dir", None)
        if not batch_size:
            logging.warning(f"No batch_size dict provided for SDXLEngineBuilderOp. Setting to default value \
                            {SDXLComponent.CLIP1: 2, SDXLComponent.CLIP2: 2, SDXLComponent.UNETXL: 2, SDXLComponent.VAE: 1}")

            batch_size = {SDXLComponent.CLIP1: 2, SDXLComponent.CLIP2: 2, SDXLComponent.UNETXL: 2, SDXLComponent.VAE: 1}

        self.config_ver = config_ver

        self.builders = []
        # Dispatch batch_size to builders
        for component, component_batch_size in batch_size.items():
            builder = SDXLEngineBuilderOp.COMPONENT_BUILDER_MAP[component](*args, component_name=component.valstr(), batch_size=component_batch_size,
                                                                           model_path=model_path, **kwargs)

            self.builders.append(builder)

    def run(self, scratch_space, dependency_outputs):
        # Build each engine separately
        for builder in self.builders:
            logging.info(f"Building {builder.model.name} from {builder.model_path}")
            if self.engine_dir is not None:
                engine_dir = Path(self.engine_dir)
            else:
                engine_dir = builder.engine_dir(scratch_space)
            engine_name = builder.engine_name(builder.device_type,
                                              builder.batch_size,
                                              builder.precision.valstr(),
                                              builder.model.name,
                                              tag=self.config_ver)
            with SDXLBuilderManager(builder) as build_manager:
                build_manager.builder(batch_size=builder.batch_size, save_to=engine_dir / engine_name)


class SDXL(LegacyBuilder):
    def __init__(self, args):
        super().__init__(SDXLEngineBuilderOp(**args))

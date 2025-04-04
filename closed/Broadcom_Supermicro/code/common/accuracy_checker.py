# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

import os
import shutil
import sys
import venv

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from importlib import import_module
from typing import List, Any, Final, Dict

from code.common import run_command, logging
from code.common.constants import *
from code.common.fix_sys_path import ScopedRestrictedImport
from code.common.systems.system_list import DETECTED_SYSTEM

# Since submission-checker uses a relative import, but we are running from main.py, we need to surface its directory
# into sys.path so it can successfully import it. Use a ScopedRestrictedImport for this.
_new_path = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, "tools", "submission")] + sys.path
with ScopedRestrictedImport(_new_path):
    submission_checker = import_module("submission_checker")
    G_ACC_PATTERNS = submission_checker.ACC_PATTERN
    # MLCommons doesn't add the current version until it is close to submission
    _submission_model_config = submission_checker.MODEL_CONFIG
    _version_str = VERSION if VERSION in _submission_model_config else "v4.0"
    G_ACC_TARGETS = _submission_model_config[_version_str]["accuracy-target"]
    G_ACC_UPPER_LIMIT = _submission_model_config[_version_str]["accuracy-upper-limit"]
    """Dict[str, Tuple[str, float]]: A dictionary mapping the benchmark name to a tuple of (accuracy_metric, threshold)"""


G_MLCOMMONS_ACC_TARGETS: Final[Dict[AccuracyTarget, str]] = {AccuracyTarget.k_99: "99",
                                                             AccuracyTarget.k_99_9: "99.9"}
"""Dict[AccuracyTarget, str]: A dictionary mapping an AccuracyTarget to the string used in MLCommons keys"""


def get_pythonpath(pythonpath_extra: Optional[str] = None) -> str:
    """Gets the PYTHONPATH value used by ScopedRestrictedImport. In other words, gets the PYTHONPATH with all user
    package directories removed.

    Args:
        pythonpath_extra (str): If set, will be prepended to the PYTHONPATH. (Default: None)

    Returns:
        str: The PYTHONPATH value
    """
    with ScopedRestrictedImport() as sri:
        pythonpath = sri.path_as_string()
    if pythonpath_extra is not None:
        # TODO: Currently this method is only called once, but this is enforced by the programmer and is prone to bugs.
        # This should filter out the final PYTHONPATH for duplicate paths while preserving order.
        pythonpath = ":".join((pythonpath_extra, pythonpath))
    return pythonpath


@dataclass
class _AccuracyScriptCommand:
    """Contains metadata for the command to invoke an MLCommons Inference accuracy script"""

    executable: str
    """str: The executable name to run. Python accuracy scripts should NOT be invoked directly (i.e. ./path/to/script.py
            via the shebang). For Python-based accuracy scripts, this value should always be "python", "python3", or
            "python3.8".
    """

    argv: List[str]
    """List[str]: List of arguments to pass to the executable. For Python scripts, this should be sys.argv."""

    env: Dict[str, str]
    """Dict[str]: Dictionary of custom environment variables to pass to the executable."""

    def __str__(self) -> str:
        argv_str = " ".join(self.argv)
        s = f"{self.executable} {argv_str}"
        if len(self.env) > 0:
            env_str = " ".join(f"{k}={v}" for k, v in self.env.items())
            s = env_str + " " + s
        return s

    def fix_pythonpath(self):
        """
        For commands in the form "python(3|3.*) *", sets PYTHONPATH to use an import path that does not include ~/.local.
        For non-Python commands, this method is a no-op.

        By default, Python will automatically import [site.py](https://docs.python.org/3/library/site.html) when the
        Python session is initialized. This is what inserts ~/.local/lib/python*/site-packages into sys.path
        automatically.  However, this can by bypassed by using the -S flag. In doing so, we will need to add the system
        site-packages directories manually by generating a string and using PYTHONPATH.

        The reason this is not done within the Makefile is because it needs to be done at every invocation of Python
        within the Makefile, and also does not fix the problem with a subprocess is called (since the subprocess ALSO
        needs to pass in -S), and does not fix the issue when users run scripts by calling Python natively, without
        using Make.
        """
        if not self.executable.startswith("python"):
            return

        # Add the -S flag to disable site.py auto-import
        if " -S" not in self.executable:
            self.executable = f"{self.executable} -S"

        # Set PYTHONPATH in the environment vars
        self.env["PYTHONPATH"] = get_pythonpath(pythonpath_extra=self.env.get("PYTHONPATH", None))


class AccuracyChecker(ABC):
    """Utility class to run a particular accuracy script from the MLCommons Inference repo.
    """

    def __init__(self,
                 log_file: str,
                 benchmark_conf: Dict[str, Any],
                 full_benchmark_name: str,
                 mlcommons_module_path: str):
        """Creates an AccuracyChecker

        Args:
            log_file (str): Path to the accuracy log
            benchmark_conf (Dict[str, Any]): The benchmark configuration used to generate the accuracy result
            full_benchmark_name (str): The full submission name of the benchmark
            mlcommons_module_path (str): The relative filepath of the accuracy script in the MLCommons Inference repo
        """
        self.log_file = log_file
        self.benchmark_conf = benchmark_conf
        self.benchmark = self.benchmark_conf["benchmark"]
        self.full_benchmark_name = full_benchmark_name
        self.mlcommons_module_path = mlcommons_module_path
        self.acc_metric_list = list(G_ACC_TARGETS[self.full_benchmark_name])[::2]
        self.threshold_list = list(G_ACC_TARGETS[self.full_benchmark_name])[1::2]
        self.acc_pattern_list = [G_ACC_PATTERNS[acc_metric] for acc_metric in self.acc_metric_list]

    @abstractmethod
    def get_cmd(self) -> _AccuracyScriptCommand:
        """Constructs the command to run the accuracy script

        Returns:
            _AccuracyScriptCommand: The command to run
        """
        raise NotImplemented

    def run(self) -> List[str]:
        """Runs the accuracy checker script and returns the output if the script ran successfully.
        """
        cmd = self.get_cmd()
        cmd.fix_pythonpath()

        return run_command(str(cmd), get_output=True)

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracies the accuracy results.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            threshold = self.threshold_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            if accuracy is None:
                print(f"WARNING: Couldn't find accuracy for {self.acc_metric_list[i]}.")

            passed = accuracy is not None and accuracy >= threshold
            accuracy_result_list.append({"name": self.acc_metric_list[i], "value": accuracy, "threshold": threshold, "pass": passed})
        return accuracy_result_list


class ResNet50AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "resnet",
                         "vision/classification_and_detection/tools/accuracy-imagenet.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                "--imagenet-val-file data_maps/imagenet/val_map.txt",
                "--dtype int32"]
        return _AccuracyScriptCommand("python3", argv, dict())


class RetinanetAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "retinanet",
                         "vision/classification_and_detection/tools/accuracy-openimages.py")
        self.openimages_dir = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "open-images-v6-mlperf")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--openimages-dir {self.openimages_dir}",
                "--output-file build/retinanet-results.json"]
        return _AccuracyScriptCommand("python3", argv, dict())


class BERTAccuracyChecker(AccuracyChecker):
    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"}  # Use FP16 output for INT8 mode
    """Dict[str, str]: Remap MLPINF precision strings to a string that the BERT accuracy script understands"""

    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"bert-{_acc_target}",
                         "language/bert/accuracy-squad.py")
        self.squad_path = os.path.join(os.environ.get("DATA_DIR", "build/data"), "squad", "dev-v1.1.json")
        self.vocab_file_path = "build/data/squad/vocab.txt" if 'CPU' in self.benchmark_conf['config_name'] else "build/models/bert/vocab.txt"
        self.output_prediction_path = os.path.join(os.path.dirname(self.log_file), "predictions.json")

        _dtype = self.benchmark_conf["precision"].lower()
        self.dtype = BERTAccuracyChecker.dtype_expand_map.get(_dtype, _dtype)

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing tokenizers on SoC systems. Use custom BERT accuracy script.
        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            argv = ["code/bert/tensorrt/accuracy-bert.py",
                    f"--mlperf-accuracy-file {self.log_file}",
                    f"--squad-val-file {self.squad_path}"]
            env = dict()
        else:
            argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                    f"--log_file {self.log_file}",
                    f"--vocab_file {self.vocab_file_path}",
                    f"--val_data {self.squad_path}",
                    f"--out_file {self.output_prediction_path}",
                    f"--output_dtype {self.dtype}"]
            env = {"PYTHONPATH": "code/bert/tensorrt/helpers"}
        return _AccuracyScriptCommand("python3", argv, env)


def validate_hf_checkpoint(checkpoint_dir: str):
    """Check if the checkpoint directory is a valid Hugging Face checkpoint.
    Raise an error if the checkpoint is not valid.
    """
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_dir, file)):
            raise ValueError(f"Missing Checkpoint in: {checkpoint_dir}. Please download or move the checkpoint to the directory.")


class GPTJAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"gptj-{_acc_target}",
                         "language/gpt-j/evaluation.py")
        self.cnn_daily_mail_path = os.path.join(os.environ.get("DATA_DIR", "build/data"), "cnn-daily-mail", "cnn_eval.json")

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing tokenizers on SoC systems. Use custom BERT accuracy script.
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.cnn_daily_mail_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


class Llama2AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"llama2-70b-{_acc_target}",
                         "language/llama2-70b/evaluate-accuracy.py")
        # Check if the local model is available for faster loading.
        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.ref_acc_pkl_path = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
                                             "open_orca", "open_orca_gpt4_tokenized_llama.sampled_24576.pkl")
        self.llama2_70b_ckpt_dir = os.path.join(os.environ.get("MODEL_DIR", "build/models"),
                                                "Llama2", "Llama-2-70b-chat-hf")
        local_model_path = Path("/raid/data/mlperf-llm/Llama-2-70b-chat-hf")
        if local_model_path.exists():
            logging.info(f"using local Llama2 model from {local_model_path}")
            self.llama2_70b_ckpt_dir = str(local_model_path)

        validate_hf_checkpoint(self.llama2_70b_ckpt_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing tokenizers on SoC systems. Use custom BERT accuracy script.
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--checkpoint-path {self.llama2_70b_ckpt_dir}",
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.ref_acc_pkl_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


class Llama3_1AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         f"llama3.1-405b",
                         "language/llama3.1-405b/evaluate-accuracy.py")
        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.dataset_path = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "llama3.1-405b", "mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl")
        self.checkpoint_dir = os.path.join(os.environ.get("MODEL_DIR", "build/models"), "Llama3.1-405B", "Meta-Llama-3.1-405B-Instruct")

        if (local_model_path := Path("/raid/data/mlperf/llm-large/Meta-Llama-3.1-405B-Instruct")).exists():
            logging.info(f"using local Llama3.1 model from {local_model_path}")
            self.checkpoint_dir = str(local_model_path)

        validate_hf_checkpoint(self.checkpoint_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--checkpoint-path {self.checkpoint_dir}",
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.dataset_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


class Mixtral8x7bAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         f"mixtral-8x7b",
                         "language/mixtral-8x7b/evaluate-accuracy.py")
        self.ref_acc_pkl_path = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "moe", "mlperf_mixtral8x7b_moe_dataset_15k.pkl")
        self.mixtral_8x7b_ckpt_dir = os.path.join(os.environ.get("MODEL_DIR", "build/models"), "Mixtral", "Mixtral-8x7B-Instruct-v0.1")
        self.upper_limit_dict = dict(zip(G_ACC_UPPER_LIMIT[self.full_benchmark_name][0::2], G_ACC_UPPER_LIMIT[self.full_benchmark_name][1::2]))

        validate_hf_checkpoint(self.mixtral_8x7b_ckpt_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [
            f"--module-path={os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path)}",
            f"--checkpoint-path={self.mixtral_8x7b_ckpt_dir}",
            f"--mlperf-accuracy-file={self.log_file}",
            f"--dataset-file={self.ref_acc_pkl_path}",
        ]

        env = dict()
        return _AccuracyScriptCommand(f"/work/code/mixtral-8x7b/tensorrt/run_accuracy.sh", argv, env)

    def get_accuracy(self) -> List[Dict[Any, Any]]:
        """Runs the accuracy script and get_accuracys the accuracy results for Mixtral-8x7B.
           Mixtral-8x7B needs to check both the lower bound and the upper bound of TOKENS_PER_SAMPLE.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "upper_limit": Float value representing the maximum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            acc_metric = self.acc_metric_list[i]
            threshold = self.threshold_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            upper_limit = self.upper_limit_dict.get(acc_metric, accuracy)
            passed = accuracy is not None and threshold <= accuracy <= upper_limit
            accuracy_result_list.append({
                "name": self.acc_metric_list[i],
                "value": accuracy,
                "threshold": threshold,
                "pass": passed,
            })

            if acc_metric in self.upper_limit_dict:
                accuracy_result_list[-1]['upper_limit'] = upper_limit

        return accuracy_result_list


class DLRMv2AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"dlrm-v2-{_acc_target}",
                         "recommendation/dlrm_v2/pytorch/tools/accuracy-dlrm.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                "--day-23-file /home/mlperf_inf_dlrmv2/criteo/day23/raw_data",
                "--aggregation-trace-file /home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.txt",
                "--dtype float32"]
        return _AccuracyScriptCommand("python3", argv, dict())


class UNET3DAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"3d-unet-{_acc_target}",
                         None)

        postprocess_dir = "build/postprocessed_data"
        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = ["code/3d-unet/tensorrt/accuracy_kits.py",
                f"--log_file {self.log_file}"]
        env = dict()
        # WAR for numpy linargerror_eigenvalues_nonconvergence happen in ARM64 (x86 don't see this)
        if DETECTED_SYSTEM.cpu.architecture == CPUArchitecture.aarch64:
            env["OPENBLAS_CORETYPE"] = "armv8"
        return _AccuracyScriptCommand("python3", argv, env)

    def run(self) -> List[str]:
        return super().run()


class RGATAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "rgat",
                         "graph/R-GAT/tools/accuracy_igbh.py")

        # Set up temporary directories
        dst = Path("/home/mlperf_inf_rgat/acc_checker")

        node_file = dst / "full" / "processed" / "paper" / "node_label_2K.npy"
        if not node_file.exists():
            node_file.parent.mkdir(parents=True, exist_ok=True)
            src = Path("/home/mlperf_inf_rgat/optimized/converted/graph/full/node_label_2K.npy")
            shutil.copy(src, node_file)

        val_index = dst / "full" / "processed" / "val_idx.pt"
        if not val_index.exists():
            val_index.parent.mkdir(parents=True, exist_ok=True)
            src = Path("/home/mlperf_inf_rgat/optimized/converted/graph/full/val_idx.pt")
            shutil.copy(src, val_index)

        self.acc_file_root = dst
        self.tmp_file = "/tmp/rgat_acc_results.txt"

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                "--dataset-size full",
                "--no-memmap",
                f"--dataset-path {self.acc_file_root}",
                f"--output-file {self.tmp_file}",
                "--dtype int64"]
        return _AccuracyScriptCommand("python3", argv, dict())

    def run(self) -> List[str]:
        super().run()
        with open(self.tmp_file, 'r') as f:
            lines = f.readlines()
        return lines

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracies the accuracy results.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        d = super().get_accuracy()[0]
        d["value"] = d["value"] / 100
        d["pass"] = (d["value"] >= d["threshold"])
        return [d]


class SDXLAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "stable-diffusion-xl",
                         "text_to_image/tools/accuracy_coco.py")
        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.sdxl_accuracy_venv_path = "/work/.sdxl-accuracy"
        self.prepare_virtual_env()
        self.compliance_image_path = os.path.join(os.path.dirname(log_file), "images")

    def run(self) -> List[str]:
        """Runs the accuracy checker script and returns the output if the script ran successfully.
        """
        cmd = self.get_cmd()
        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) < 100:
            cmd.fix_pythonpath()

        return run_command(str(cmd), get_output=True)

    def prepare_virtual_env(self):
        """SDXL accuracy script is not compatible with mlpinf container, the accuracy checker installs the virtual env
           with the same python dependencies as the reference implementation
        """

        if os.path.exists(self.sdxl_accuracy_venv_path):
            logging.warning(f"SDXL accuracy virtual env exists, bypassing SDXLAccuracyChecker.prepare_virtual_env()")
        else:
            venv.create(self.sdxl_accuracy_venv_path, with_pip=True)
            run_command(f"{self.sdxl_accuracy_venv_path}/bin/pip install -r /work/code/stable-diffusion-xl/tensorrt/accuracy_requirements.txt")

    def get_cmd(self) -> _AccuracyScriptCommand:
        statistics_path = os.path.join(G_MLCOMMONS_INF_REPO_PATH, "text_to_image/tools/val2014.npz")
        caption_path = os.path.join(G_MLCOMMONS_INF_REPO_PATH, "text_to_image/coco2014/captions/captions_source.tsv")
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--caption-path {caption_path}",
                f"--statistics-path {statistics_path}",
                f"--output-file /tmp/sdxl-accuracy.json",
                f"--compliance-images-path {self.compliance_image_path}",
                "--device gpu" if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) < 100 else "--device cpu"]

        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            argv.append("--low_memory")

        return _AccuracyScriptCommand(f"{self.sdxl_accuracy_venv_path}/bin/python3", argv, dict())

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracys the accuracy results for SDXL.
           SDXL needs to check both the lower bound and the upper bound of FID and CLIP

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "upper_limit": Float value representing the maximum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            threshold = self.threshold_list[i]
            upper_limit = self.upper_limit_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            passed = accuracy is not None and accuracy >= threshold and accuracy <= upper_limit
            accuracy_result_list.append({"name": self.acc_metric_list[i], "value": accuracy, "threshold": threshold, "upper_limit": upper_limit, "pass": passed})
        return accuracy_result_list


G_ACCURACY_CHECKER_MAP = {Benchmark.BERT: BERTAccuracyChecker,
                          Benchmark.DLRMv2: DLRMv2AccuracyChecker,
                          Benchmark.GPTJ: GPTJAccuracyChecker,
                          Benchmark.LLAMA2: Llama2AccuracyChecker,
                          Benchmark.LLAMA2_Interactive: Llama2AccuracyChecker,
                          Benchmark.LLAMA3_1: Llama3_1AccuracyChecker,
                          Benchmark.Mixtral8x7B: Mixtral8x7bAccuracyChecker,
                          Benchmark.ResNet50: ResNet50AccuracyChecker,
                          Benchmark.Retinanet: RetinanetAccuracyChecker,
                          Benchmark.UNET3D: UNET3DAccuracyChecker,
                          Benchmark.RGAT: RGATAccuracyChecker,
                          Benchmark.SDXL: SDXLAccuracyChecker}
"""Dict[Benchmark, AccuracyChecker]: Maps a Benchmark to its AccuracyChecker"""


def check_accuracy(log_file, config):
    """Check accuracy of given benchmark."""
    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."

    # Check if log_file is empty by just reading first several bytes
    # The first 4B~6B is likely all we need to check: '', '[]', '[]\r', '[\n]\n', '[\r\n]\r\n', ...
    # but checking 8B for safety
    with open(log_file, 'r') as lf:
        first_8B = lf.read(8)
        if not first_8B or ('[' in first_8B and ']' in first_8B):
            return "No accuracy results in PerformanceOnly mode."

    benchmark = config["benchmark"]
    accuracy_checker = (G_ACCURACY_CHECKER_MAP.get(benchmark, None))(log_file, config)  # Create an instance
    if accuracy_checker is None:
        raise ValueError(f"Invalid benchmark {benchmark} does not have an AccuracyChecker.")

    return accuracy_checker.get_accuracy()

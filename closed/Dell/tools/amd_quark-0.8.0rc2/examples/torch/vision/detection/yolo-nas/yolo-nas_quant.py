#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import torch
import itertools
import super_gradients
import super_gradients.training.training_hyperparams
from super_gradients.training.dataloaders import coco2017_val_yolo_nas, coco2017_train_yolo_nas
import torch.optim
import torch.fx
from torch.utils.data import DataLoader
import numpy as np
from torch._export import capture_pre_autograd_graph
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='yolo-nas', choices=['yolo-nas'], help='Model to use.')
parser.add_argument('--data_dir',
                    default='{DATA_PATH}/COCO_dataset',
                    help='Data set directory.')
parser.add_argument('--pretrained', default=None, help='Pre trained model weights')
parser.add_argument('--qat', action="store_true", help='Perform QAT to further improve accuracy.')
parser.add_argument('--train_batch_size', default=96, type=int, help='Batch size for training.')
parser.add_argument('--train_data_percent', default=0.008, type=float, help='Percentage for using full Dataset.')
parser.add_argument('--val_batch_size', default=32, type=int, help='Batch size for validation.')
parser.add_argument('--calib_data_size', default=10, type=int, help='Data used for calibration.')
parser.add_argument('--gpus', type=str, default='0', help='gpu ids to be used for training, seperated by commas')
parser.add_argument("--model_export",
                    help="Model export format",
                    default=['onnx'],
                    action="append",
                    choices=[None, "torch_compile", "'onnx'", "torch_save"])
parser.add_argument('--export_dir', default="./quant_result", help='Dir to save export model.')
args, _ = parser.parse_known_args()


def get_val_data_loader(percentage: float = 1.0):
    valid_dataloader = coco2017_val_yolo_nas(dataloader_params={"batch_size": args.val_batch_size},
                                             dataset_params={"data_dir": args.data_dir})
    dataset_size = len(valid_dataloader.dataset)
    indices = list(range(dataset_size))
    split = int(dataset_size * percentage)
    if split != dataset_size:
        print("Validation Dataloader size from {} to {}".format(dataset_size, split))
    new_datast = torch.utils.data.Subset(valid_dataloader.dataset, indices[:split])
    new_data_loader = DataLoader(dataset=new_datast, **valid_dataloader.dataloader_params)
    valid_dataloader = new_data_loader
    return valid_dataloader


def get_train_data_loader(percentage: float = 0.06):
    train_dataloader = coco2017_train_yolo_nas(dataloader_params={"batch_size": args.train_batch_size},
                                               dataset_params={"data_dir": args.data_dir})
    dataset_size = len(train_dataloader.dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(dataset_size * percentage)
    print("Train Dataloader size from {} to {}".format(dataset_size, split))
    new_datast = torch.utils.data.Subset(train_dataloader.dataset, indices[:split])
    new_data_loader = DataLoader(dataset=new_datast, **train_dataloader.dataloader_params)
    train_dataloader = new_data_loader
    return train_dataloader


def val_coco(model, dataset=None):
    '''
    COCO 2017 Dataset:
    - Download coco dataset:
        annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        train2017: http://images.cocodataset.org/zips/train2017.zip
        val2017: http://images.cocodataset.org/zips/val2017.zip

    - Unzip and organize it as below:
        coco
        ├── annotations
        │      ├─ instances_train2017.json
        │      ├─ instances_val2017.json
        │      └─ ...
        └── images
            ├── train2017
            │   ├─ 000000000001.jpg
            │   └─ ...
            └── val2017
                └─ ...
    '''
    torch.cuda.empty_cache()
    valid_dataloader = coco2017_val_yolo_nas(dataloader_params={"batch_size": args.val_batch_size},
                                             dataset_params={"data_dir": args.data_dir})
    # new_parameter = v
    trainer = Trainer("yolo_nas_experiment")
    metric1 = DetectionMetrics_050(score_thres=0.1,
                                   top_k_predictions=300,
                                   num_cls=80,
                                   normalize_targets=True,
                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                          nms_top_k=1000,
                                                                                          max_predictions=300,
                                                                                          nms_threshold=0.7))
    metric2 = DetectionMetrics_050_095(score_thres=0.1,
                                       normalize_targets=True,
                                       top_k_predictions=300,
                                       num_cls=80,
                                       post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                              nms_top_k=1000,
                                                                                              max_predictions=300,
                                                                                              nms_threshold=0.7))

    result = trainer.test(
        model.eval(),
        test_loader=valid_dataloader,
        test_metrics_list=[metric1, metric2],
    )
    print(result)
    torch.cuda.empty_cache()
    return result


def updadate_config(train_config):
    train_config['warmup_initial_lr'] = 1e-7
    train_config['initial_lr'] = 2e-7
    train_config['cosine_final_lr_ratio']
    train_config['lr_warmup_epochs'] = 1
    train_config['lr_warmup_steps'] = 8
    train_config['max_epochs'] = 1


def train_model(model):
    valid_dataloader = get_val_data_loader()
    train_dataloader = get_train_data_loader(args.train_data_percent)
    # train parameters NOTE this is just for demonstration
    train_config = super_gradients.training.training_hyperparams.get("coco2017_yolo_nas_s")
    updadate_config(train_config)
    trainer = Trainer(experiment_name="yolo_nas_s", ckpt_root_dir="CHECKPOINT_DIR")
    trainer.train(model=model,
                  training_params=train_config,
                  train_loader=train_dataloader,
                  valid_loader=valid_dataloader)
    torch.cuda.empty_cache()
    return model


def main():
    print('Used arguments:', args)
    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"
    if device_ids is None:
        device = 'cpu'

    # Parpare the model and dataset
    dummy_input = torch.ones(1, 3, 640, 640, requires_grad=False).to(device=device)
    valid_dataloader = coco2017_val_yolo_nas(dataloader_params={"batch_size": 25},
                                             dataset_params={"data_dir": args.data_dir})

    calib_data = [x[0].to(device) for x in list(itertools.islice(valid_dataloader, args.calib_data_size))]

    yolo_nas = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").to(device).eval()
    # TODO explane the reason
    yolo_nas.prep_model_for_conversion(input_size=[1, 3, 640, 640])
    graph_model = capture_pre_autograd_graph(yolo_nas.eval(), (dummy_input, ))
    # NOTE  In the future, we will use torch.export.export_for_training to replace capture_pre_autograd_graph

    # Init quantization config and let Quantizer automatically insert quantizer
    INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)
    quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                      output_tensors=INT8_PER_TENSOR_SPEC,
                                      weight=INT8_PER_TENSOR_SPEC,
                                      bias=INT8_PER_TENSOR_SPEC)
    # NOTE this is a temponary config
    exclude_node = [
        '_param_constant310', '_set_grad_enabled_1', '_param_constant310', 'softmax', '_param_constant323', 'softmax_1',
        '_param_constant336', 'softmax_2'
    ]  # [1671-1938] = [1671-1717] [1722, 1768], [1773, 1819]
    quant_config = Config(global_quant_config=quant_config,
                          quant_mode=QuantizationMode.fx_graph_mode,
                          exclude=exclude_node)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, calib_data)
    graph_model = None
    if args.qat is True:
        train_model(quantized_model)
    # export session
    if args.model_export is not None:
        freezeded_model = quantizer.freeze(quantized_model.eval())
        val_coco(freezeded_model)
        if "onnx" in args.model_export:
            from quark.torch import ModelExporter
            from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
            config = ExporterConfig(json_export_config=JsonExporterConfig())
            exporter = ModelExporter(config=config, export_dir=args.export_dir)
            example_inputs = (torch.rand(args.val_batch_size, 3, 640, 640).to(device), )
            exporter.export_onnx_model(freezeded_model, example_inputs[0])
        if "torch_save" in args.model_export:
            from quark.torch.export.api import save_params
            example_inputs = (dummy_input, )
            save_params(freezeded_model,
                        model_type=args.model_name,
                        args=example_inputs,
                        export_dir=args.export_dir,
                        quant_mode=quant_config.quant_mode)
        if "torch_compile" in args.model_export:
            print("\nCalling PyTorch 2 torch.compile...")
            # Note: The model after torch.compile may not be able to export to other format
            freezeded_model = torch.compile(freezeded_model)


if __name__ == '__main__':
    main()

Supported Data and Op Types
===========================

Supported Data Types
--------------------

Summary Table
~~~~~~~~~~~~~

+---------------------+
| Supported Data Types|
+=====================+
| Int4 / Uint4        |
+---------------------+
| Int8 / Uint8        |
+---------------------+
| Int16 / Uint16      |
+---------------------+
| Int32 / Uint32      |
+---------------------+
| Float16             |
+---------------------+
| BFloat16            |
+---------------------+
| BFP16               |
+---------------------+

.. note::
    
   When installing on Windows, Visual Studio is required. The minimum version of Visual Studio is Visual Studio 2022. During the compilation process, there are two ways to use it:  
  
1. **Use the Developer Command Prompt for Visual Studio**    
   When installing Visual Studio, ensure that the Developer Command Prompt for Visual Studio is installed as well. Execute programs in the CMD window of the Developer Command Prompt for Visual Studio.  
  
2. **Manually Add Paths to Environment Variables**    
   Visual Studio's ``cl.exe``, ``MSBuild.exe``, and ``link.exe`` will be used. Ensure that the paths are added to the `PATH` environment variable. These programs are located in the Visual Studio installation directory. In the *Edit Environment Variables* window, click **New**, then paste the path to the folder containing ``cl.exe``, ``link.exe``, and ``MSBuild.exe``. Click **OK** on all windows to apply the changes. 

1. Quantizing to Other Precisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the INT8/UINT8, the quark.onnx supports quantizing models to other data formats, including INT16/UINT16, INT32/UINT32, Float16 and BFloat16, which can provide better accuracy or be used for experimental purposes. These new data formats are achieved by a customized version of QuantizeLinear and DequantizeLinear named "VitisQuantizeLinear" and "VitisDequantizeLinear", which expand onnxruntime's UInt8 and Int8 quantization to support UInt16, Int16, UInt32, Int32, Float16, and
BFloat16. This customized Q/DQ was implemented by a custom operations library in quark.onnx using onnxruntime's custom operation C API.

The custom operations library was developed based on Linux and Windows.

To use this feature, the ``quant_format`` should be set to VitisQuantFormat.QDQ. You might have noticed that in both the recommended NPU_CNN and NPU_Transformer configurations, the ``quant_format`` is set to QuantFormat.QDQ. NPU targets that support acceleration for models quantized to INT8/UINT8, do not support other precisions.

.. note::
    
     When the Quant_Type is Int4/UInt4, the onnxruntime version must be 1.19.0 or higher. Only the onnxruntime native "CalibrationMethod" is supported (MinMax, Percentile), and the quant_format is required to be QuantFormat.  

1.1 Quantizing Float32 Models to Int16 or Int32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer supports quantizing float32 models to Int16 or Int32 data formats. To enable this, you need to set the ``activation_type`` and ``weight_type`` in the quantize_static API to the new data types. Options are VitisQuantType.QInt16/VitisQuantType.QUInt16 or VitisQuantType.QInt32/VitisQuantType.QUInt32.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.VitisQuantFormat.QDQ,
       activation_type=quark.onnx.VitisQuantType.QInt16,
       weight_type=quark.onnx.VitisQuantType.QInt16,
   )

1.2 Quantizing Float32 Models to Float16 or BFloat16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides integer data formats, the quantizer also supports quantizing Float32 models to Float16 or BFloat16 data formats. Set the ``activation_type`` and ``weight_type`` to ``VitisQuantType.QFloat16`` or ``VitisQuantType.QBFloat16``.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.VitisQuantFormat.QDQ,
       activation_type=quark.onnx.VitisQuantType.QFloat16,
       weight_type=quark.onnx.VitisQuantType.QFloat16,
   )

1.3 Quantizing Float32 Models to BFP16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer also supports quantizing Float32 models to BFP16 data formats. The block size can be modified by changing the ``block_size`` parameter in the ``extra_options``. Currently, BFP16 only supports symmetric activation. The following is the configuration for BFP16 with a block size of 8.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.NonOverflow,
       quant_format=quark.onnx.VitisQuantFormat.BFPFixNeuron,
       extra_options={
           "ActivationSymmetric": True,
           "BFPAttributes": {
               "bfp_method": "to_bfp",
               "bit_width": 16,
               "block_size": 8,
           }
       },
   )

.. note::
    
     When inference with ONNX Runtime, we need to register the custom op's so(Linux) or dll(Windows) file in the ORT session options.

.. code:: python

    import onnxruntime
    from quark.onnx import get_library_path as vai_lib_path

    # Also We can use the GPU configuration: 
    # device='cuda:0'
    # providers = ['CUDAExecutionProvider']

    device = 'cpu'
    providers = ['CPUExecutionProvider']

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(vai_lib_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

1.4 Quantizing Float32 Models to Mixed Data Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer even supports setting the activation and weight to different precisions. For example, activation is Int16 while weight is Int8. This can be used when pure Int8 quantization can not meet accuracy requirements.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       quant_format=quark.onnx.VitisQuantFormat.QDQ,
       activation_type=quark.onnx.VitisQuantType.QInt16,
       weight_type=QuantType.QInt8,
   )

2. Quantizing Float16 Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models in Float16, we recommend setting ``convert_fp16_to_fp32`` to True. This first converts your Float16 model to a Float32 model before quantization, reducing redundant nodes such as cast in the model.

.. code:: python

   quark.onnx.quantize_static(
       model_input,
       model_output,
       calibration_data_reader,
       quant_format=QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8,
       enable_NPU_cnn=True,
       convert_fp16_to_fp32=True,
       extra_options={'ActivationSymmetric':True}
   )

.. note::
    When using ``convert_fp16_to_fp32`` in quark.onnx, it requires onnxsim to simplify the ONNX model. Ensure that onnxsim is installed by using ``python -m pip install onnxsim``.

Supported Op Type
-----------------

.. _summary-table-1:

Summary Table
~~~~~~~~~~~~~

Table: List of Quark ONNX Supported Quantized Ops 

+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Supported Ops         | Comments                                                                                                                                                                                                  |
+=======================+===========================================================================================================================================================================================================+
| Add                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ArgMax                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| AveragePool           | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| BatchNormalization    | By default, the "optimize_model" parameter will fuse BatchNormalization to Conv/ConvTranspose/Gemm. For standalone BatchNormalization, quantization is supported only for NPU_CNN platforms by converting |
|                       | BatchNormalization to Conv.                                                                                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Clip                  | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Concat                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Conv                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ConvTranspose         |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| DepthToSpace          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Div                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Erf                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gather                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gemm                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| GlobalAveragePool     |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HardSigmoid           | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| InstanceNormalization |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LayerNormalization    | Supported for opset>=17. Will be quantized only when its input is quantized.                                                                                                                              |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LeakyRelu             |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LpNormalization       | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MatMul                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Min                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Max                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MaxPool               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Mul                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pad                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PRelu                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ReduceMean            | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Relu                  | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Reshape               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Resize                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Slice                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sigmoid               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Softmax               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SpaceToDepth          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Split                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Squeeze               | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sub                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Tanh                  | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Transpose             | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Unsqueeze             | Will be quantized only when its input is quantized.                                                                                                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Where                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

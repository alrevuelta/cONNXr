import onnx
from quantize import quantize, QuantizationMode
import numpy as np

# Load the onnx model
model = onnx.load('../test/mnist/model.onnx')

# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, static=True,
                           input_quantization_params={
                                'Input3': [np.uint8(113), np.float32(0.05)],
                                'Pooling66_Output_0': [np.uint8(113), np.float32(0.05)],
                                'Pooling160_Output_0_reshape0': [np.uint8(113), np.float32(0.05)],
                                'Parameter193_reshape1': [np.uint8(113), np.float32(0.05)],
                           },
                            output_quantization_params={
                                'Convolution28_Output_0': [np.uint8(113), np.float32(0.05)],
                                'Convolution110_Output_0': [np.uint8(113), np.float32(0.05)],
                                'Times212_Output_0': [np.uint8(113), np.float32(0.05)],
                           })
# Save the quantized model
onnx.save(quantized_model, 'model_quantized.onnx')
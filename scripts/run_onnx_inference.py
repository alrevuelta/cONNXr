import onnx
import os
import glob
import onnxruntime.backend as backend
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
from timeit import Timer
import numpy
import sys
import subprocess
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
"""
    file:        run_onnx_inference.py

    description: tool for debugging purposes. just open a model and run
    inference on it using the official onnx runtime. Some code also is
    provided to inspect the intermediate outputs of a model.
"""

test_data_dir = 'test/mobilenetv2-1.0/test_data_set_0'
model_path = 'test/mobilenetv2-1.0/mobilenetv2-1.0.onnx'
input_name = "data"

inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
print("inputs_num", inputs_num)

for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    tensor = numpy_helper.to_array(tensor)
    tensor = np.expand_dims(tensor, axis=(0, 1))
    inputs.append(tensor)
    print(tensor.dtype)
    print("tensor.shape =", tensor.shape)

inputDict = {
    input_name: inputs[0]
}

"""
Prints the intermediate outputs of a given model with extra information
like dimensions.
"""
def print_model_outputs(model_path, input_tensor, print_tensor=False):
    model_onnx = load_onnx_model(model_path)
    for idx, out in enumerate(enumerate_model_node_outputs(model_onnx)):
        print_specific_output(model_path, input_tensor, out, print_tensor)
        

def print_specific_output(model_path, input_tensor, output_name, print_tensor=False):
    model_onnx = load_onnx_model(model_path)
    num_onnx = select_model_inputs_outputs(model_onnx, output_name)
    save_onnx_model(num_onnx, "remove_temp.onnx")
    sess = rt.InferenceSession("remove_temp.onnx")
    out_tensor = sess.run(None, input_tensor)
    print("name", output_name, "shape", out_tensor[0].shape)
    if print_tensor:
        print(out_tensor[0])


#print_model_outputs(model_path, inputDict, print_tensor=False)
inputDict['data'] = inputDict['data'].reshape((1, 3, 224, 224))
print_specific_output(model_path, inputDict, 'mobilenetv20_features_relu0_fwd', print_tensor=True)
#sess = rt.InferenceSession(model_path)
#out = sess.run(None, inputDict)

#print(out[0].shape)
#print(out[0])
#print(out[0][:,:,0:5,0:5])

#print("input", inputs[0])

import onnx
import os
import glob
import onnxruntime.backend as backend
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
import time
import sys

"""
    file:        run_onnx_inference.py

    description: Given a onnx model and a set of input(s), uses the onnxruntime library
    to run inference on it. Some initial tests to benchmark the time it takes to run
    with the official onnx library, and compare against our CONNX implementation
"""

test_data_dir = '../test/mnist/test_data_set_0'

inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))

for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

inputDict = {
    "Input3": inputs[0]
}

sess = rt.InferenceSession('../test/mnist/model.onnx')

# TODO Maybe they offer profiling tools? onnxruntime.SessionOptions.enable_profiling
tic = time.time()
outputs = sess.run(None, inputDict)
toc = time.time()

print(outputs[0])
print(toc-tic)

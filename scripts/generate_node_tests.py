import onnx
import os
import glob
import onnxruntime.backend as backend
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import onnxruntime as rt
from skl2onnx import convert_sklearn
import pprint
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import Int64TensorType
import numpy as np
from onnx import numpy_helper

"""
    file:        generate_node_tests.py
    
    description: This script takes a ONNX machine learning testing model as
    input, and generates test data per node. So, lets say that your model has
    10 nodes, one input and one output. If you are using a ONNX model, that will
    come with some input/output values that are used to verify that the model
    inference is correct. Okay, so this script just creates more testing
    arrays. Lets say your model is:
    input->node[0]->node[1]->node[2]->node[3]->node[4]->output
    With ONNX test arrays, you just have an input and an expected output.
    This script generates as many models as nodes the model has. So the following
    models are generated:
    -node[0]->output
    -node[0]->node[1]->output
    -node[0]->node[1]->node[2]->output
    -...
    So this allows you to have more granularity when testing your backend, because
    you can easily identify the node (operator) that is failing
    
    see: https://github.com/onnx/models/tree/master/vision/classification/mnist
"""


def numpy_to_pb(name, np_data, out_filename):
    tensor = numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)

model_onnx = onnx.load('../test/tiny_yolov2/Model.onnx')
test_data_dir = '../test/tiny_yolov2/test_data_set_0'

inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))

for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))


"""
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    inputs.append(onnx.load_tensor(input_file))
"""

# For some models this string has to match the input to the model
inputDict = {
    "image": inputs[0]
}


outputs_list = []
for out in enumerate_model_node_outputs(model_onnx):
    outputs_list.append(out)

print(outputs_list)

for idx, out in enumerate(outputs_list):
    name = str(idx) + "_" + out
    dataset = "test_data_set_0"
    os.mkdir(name)
    os.mkdir(name + "/" + dataset)
    modelPath = name + "/" + name + ".onnx"
    model_output = select_model_inputs_outputs(model_onnx, out)
    save_onnx_model(model_output, modelPath)
    sess = rt.InferenceSession(modelPath)
    numX = sess.run(None, inputDict)
    print()
    print("Generating idx=", idx)
    print(out)
    print(numX)

    # hardcoded for 1 output
    numpy_to_pb(out, numX[0], name + "/" + dataset + "/" + "output_0.pb")
    numpy_to_pb(out, inputs[0], name + "/" + dataset + "/" + "input_0.pb")

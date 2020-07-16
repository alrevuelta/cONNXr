import onnx
import os
import glob
import onnxruntime.backend as backend
import onnxruntime as rt
from onnxruntime import RunOptions
from onnxruntime import SessionOptions
import numpy as np
from onnx import numpy_helper
import numpy
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
"""
    file:        assert_nodes.py

    description: Takes a onnx model and a cONNXr dump and asserts which node
    is not matching the outputs. Used for debugging a model that is failing.
    This script will point to the node name that is not matching.

    Under development
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

# hardcoded for mobilenetv2
inputDict['data'] = inputDict['data'].reshape((1, 3, 224, 224))


def get_node_output(model_path, input_tensor, output_name):
    temp_file = "remove_temp.onnx"
    model_onnx = load_onnx_model(model_path)
    num_onnx = select_model_inputs_outputs(model_onnx, output_name)
    save_onnx_model(num_onnx, temp_file)
    ses_opt = SessionOptions()
    # avoid warnings
    ses_opt.log_severity_level = 4
    sess = rt.InferenceSession(temp_file, sess_options=ses_opt)
    out_tensor = sess.run(None, input_tensor)
    os.remove(temp_file)
    return out_tensor[0]

# convert nested list into 1 dim
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

with open("dump.txt", 'r') as dump_file:
    lines = dump_file.readlines()
    index = 0
    for name, shape, tensor in zip(lines[0::3], lines[1::3], lines[2::3]):
        name = name.replace("name=", "").strip()
        shape = shape.replace("shape=", "").split(",")[:-1]    #remove last empty element
        tensor = tensor.replace("tensor=", "").split(",")[:-1] #remove last empty element

        # cast values
        shape = [int(i) for i in shape]
        tensor = [float(i) for i in tensor]

        n_values_to_assert = len(tensor)

        expected_tensor_np = get_node_output(model_path, inputDict, name)
        expected_tensor = flatten(expected_tensor_np.tolist())

        print("Input tensor size:", len(tensor), "Expected tensor size:", len(expected_tensor))
        print("Input tensor shape:", shape, "Expected tensor shape:", expected_tensor_np.shape)

        error = False
        for i in range(n_values_to_assert):
            if (tensor[i] - expected_tensor[i]) > 0.001:
                print("Error", tensor[i], expected_tensor[i])
                error = True

        if error:
            print("Index", index, "There was an error in", name)
            raise Exception()
        else:
            print("Index", index, "Output", name, "...asserted ok")

        index += 1




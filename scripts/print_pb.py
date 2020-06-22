import onnx
import sys
import numpy
from onnx import numpy_helper

"""
    file:        print_pb.py
    
    description: Just some utilities to print .pb files
    
    see:
"""

def print_pb_file(filename):
    tensor   = onnx.load_tensor(filename)
    print(tensor.name)
    print(tensor.dims)
    print(len(tensor.raw_data))
    print(dir(tensor))
    #tensor.dims[:] = [672, 672]
    np_array = numpy_helper.to_array(tensor)
    print("Name:", tensor.name)
    print("Data Type:", tensor.data_type)
    print("Shape:", np_array.shape)
    print(np_array)


def numpy_to_pb(name, np_data, out_filename):
    tensor = numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)


if __name__ == '__main__':
    numpy.set_printoptions(threshold=sys.maxsize)
    #print_pb_file(
    #    "test/super_resolution/test_data_set_0/input_0.pb")
    #print_pb_file(
    #    "test/super_resolution/test_data_set_0/output_0.pb")

    tensor   = onnx.load_tensor("test/super_resolution/test_data_set_0/output_0.pb")
    tensor.dims[:] = [1, 1, 672, 672]
    onnx.save_tensor(tensor, "test/super_resolution/test_data_set_0/output_0_fixed.pb")


    #print_pb_file(
    #    "../test/node/test_maxpool_2d_same_upper/test_data_set_0/output_0.pb")

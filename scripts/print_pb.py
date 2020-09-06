import onnx
import sys
import numpy
import argparse
from onnx import numpy_helper

"""
    file:        print_pb.py
    
    description: Just some utilities to print .pb files
    
    see:
"""

def print_pb_file(filename):
    tensor = onnx.load_tensor(filename)
    print("Name", tensor.name)
    print("Dimsims", tensor.dims)
    #print(len(tensor.raw_data))
    #print(dir(tensor))
    np_array = numpy_helper.to_array(tensor)
    print("Data Type:", tensor.data_type)
    print("Shape:", np_array.shape)
    print("Values:", np_array)


def numpy_to_pb(name, np_data, out_filename):
    tensor = numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    numpy.set_printoptions(threshold=sys.maxsize)
    print("Print pb with path", args.file)
    print_pb_file(args.file)

from ctypes import *
import unittest

connxr = cdll.LoadLibrary('build/libconnxr.so')

test_model_function = connxr.test_model
test_model_function.restype = c_double

class TestModel():
    def test_model(self):
        for testvector in self.io:
            result = test_model_function(self.id.encode('UTF-8'),
                                         self.path.encode('UTF-8'),
                                         testvector.encode('UTF-8'),
                                         self.n_inputs, self.n_outputs)
            if result >= 0:
                print("Inference time", result)
            else:
                print("Error, inference time < 0")
            # TODO Workaround for windows ci. Looks like time measurement
            # doesnt work well, and in some time mnist model execution time
            # is measured as 0 sec
            self.assertTrue(result >= 0)

class TestMnist(TestModel, unittest.TestCase):
    id = "mnist"
    path = "test/mnist/model.onnx"
    io = ["test/mnist/test_data_set_0"]
    n_inputs = 1
    n_outputs = 1


class TestMobilenetv2(TestModel, unittest.TestCase):
    id = "mobilenetv2"
    path = "test/mobilenetv2-1.0/mobilenetv2-1.0.onnx"
    io = ["test/mobilenetv2-1.0/test_data_set_0"]
    n_inputs = 1
    n_outputs = 1


class TestSuperresolution(TestModel, unittest.TestCase):
    id = "super_resolution"
    path = "test/super_resolution/super_resolution.onnx"
    io = ["test/super_resolution/test_data_set_0"]
    n_inputs = 1
    n_outputs = 1


class TestTinyyolov2(TestModel, unittest.TestCase):
    id = "tinyyolov2"
    path = "test/tiny_yolov2/Model.onnx"
    io = ["test/tiny_yolov2/test_data_set_0"]
    n_inputs = 1
    n_outputs = 1


if __name__ == "__main__":
    print("Testing models:")
    unittest.main(verbosity=3)

from ctypes import *
import unittest

connxr = CDLL('build/libconnxr.so')


def test_model(model_id, model_path, io_path, n_inputs, n_outputs):
    test_model_function = connxr.test_model
    test_model_function.restype = c_double
    result = test_model_function(model_id, model_path, io_path, n_inputs, n_outputs)
    if result > 0:
        print("Inference time", result)
    return result


class TestModels(unittest.TestCase):
    def test_mnist(self):
        self.assertTrue(test_model(b"mnist",
                                   b"test/mnist/model.onnx",
                                   b"test/mnist/test_data_set_0",
                                   1, 1) > 0)

        self.assertTrue(test_model(b"mnist",
                                   b"test/mnist/model.onnx",
                                   b"test/mnist/test_data_set_1",
                                   1, 1) > 0);
        self.assertTrue(test_model(b"mnist",
                                   b"test/mnist/model.onnx",
                                   b"test/mnist/test_data_set_2",
                                   1, 1) > 0);

    def test_mobilenetv2(self):
        self.assertTrue(test_model(b"mobilenetv2",
                                   b"test/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
                                   b"test/mobilenetv2-1.0/test_data_set_0",
                                   1, 1) > 0);

    def test_super_resolution(self):
        self.assertTrue(test_model(b"super_resolution",
                                   b"test/super_resolution/super_resolution.onnx",
                                   b"test/super_resolution/test_data_set_0",
                                   1, 1) > 0);

    def test_tinyyolov2(self):
        self.assertTrue(test_model(b"tinyyolov2",
                                   b"test/tiny_yolov2/Model.onnx",
                                   b"test/tiny_yolov2/test_data_set_0",
                                   1, 1) > 0);


if __name__ == "__main__":
    print("Testing models:")
    unittest.main(verbosity=3)

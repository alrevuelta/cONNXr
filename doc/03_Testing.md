# 03 Testing

Everything is tested using `cunit` framework. Note that `onnx` provides data for testing a backend implementation. Have a look to https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md and https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node. In this link you will find sets of inputs/outputs + a model.onnx that allows to test both an individual operator or a whole model, hence we divide two different types of tests:
* Operators Tests: Tests an individual operator. Lets see a simple example. We want to test the operator `Sum` over a one dimension array. So we provide a set of two inputs `X1 = [1 1]` and `X2 = [2 2]` and an expected output `O = [3 3]`. We also provides a model with just one two inputs an a node which operator is `Sum`. All these inputs/outputs
* Model Tests: Runs inference using a model. As an example we can use MNIST model (a digit recognition library). The input is an image with a written digit and the output is an int `[0-9]`. This type of tests run several operators.

You can run all the tests with
```
make test
```

If you are implementing a new operator or debugging, you might want to run only one specific test. The following command will run a specific testcase from its test suite:
```
make all ts=onnxBackendSuite tc=test_operator_maxpool_1d_default
```

## Operator Tests
// TODO List the testcases that are passing. Automate?

## Model Tests

Models that are being tested.
* https://github.com/onnx/models/tree/master/vision/classification/mnist
* https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2
* TODO: https://github.com/onnx/models/tree/master/vision/classification/resnet

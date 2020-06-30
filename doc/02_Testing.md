# 03 Testing

Everything is tested using `cunit` framework. Note that `onnx` provides data for testing a backend implementation. Have a look to https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md and https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node. In this link you will find sets of inputs/outputs + a model.onnx that allows to test both an individual operator or a whole model, hence we divide two different types of tests:
* Operator Tests: Tests an individual operator. Lets see a simple example. We want to test the operator `Sum` over a one dimension array. So we provide a set of two inputs `X1 = [1 1]` and `X2 = [2 2]` and an expected output `O = [3 3]`. We also provides a model with just one two inputs an a node which operator is `Sum`. All these inputs/outputs
* Model Tests: Runs inference using a model. As an example we can use MNIST model (a digit recognition library). The input is an image with a written digit and the output is an int `[0-9]`. This type of tests run several operators.

## Operator Tests
You can run all operator tests with the following command. This will run all the tests that are uncommented in `tests.c`. Make sure you have compiled before with `make all`.

```
make test_operators
```

You can run a single test as follows:
```
# Not working?
make test_operators OPERATORS=test_operator_maxpool_1d_default
```

## Model Tests
Different end to end models are implemented and tested, and you can run them all with the following command (make sure you can compiled before `make all`).
```
make test_models
```

You can also run a single model with `MODELS=`. In this case we run the `mnist` model.
```
make test_models MODELS=mnist
```

# Embedded Machine Learning C ONNX Runtime
Note that this project is in a very early stage so its not production ready yet. Developers are needed so feel free to contact or contribute with a pull request. Use at your own risk.

`embeddedml` provides a `C` runtime that uses `onnx` (Open Neural Network Exchange) models. With this library you can run machine learning inference in pure `C` code, that can compile with `C89` standard. In other words, given an input and a previously trained model, this tool will allow you to predict the output. This project is aimed to fill the gap between `onnx` and small embedded devices, that don't have many resources and where is not possible to use modern compilers. Just train your model with your favourite tool (sk-learn, keras, tf, pytorch), export the `model.onnx` with onnx and use it to feed `embeddedml`. As simple as that.

# High level requirements

* A pure C runtime for onnx shall be developed
* No external libraries shall be used
* Code shall compile with `C89` standard
* Compiled code shall be as little as possible (i.e. fit into an Arduino)

# Current limitations

* The graph should have a set of nodes connected in cascade. So the output of one node is always the input of the following.
* Very few basic operators are implemented.
* Onnx supports many types (`fixed-point`, `float`, `int`,...). Few of them are implemented.

# Run

You will find a simple `Makefile` inside `src`. This has been tested only in Mac so far. Not that you have to link against `-lprotobuf-c` library, that takes care of reading the `protocol-buffers` that are used to store the `onnx` file.

```
cd src
make main
```

# Protocol Buffers
`onnx` uses protocol buffers to serialize the models data. Note that `protobuf-c` is used to generate the `onnx.pb-c.c` and `onnx.pb-c.h`. Files are already provided, but you can generate it like this:

```
protoc --c_out=. onnx.proto
```

# Tests
`cunit` is used to test the code.

* Have a look to https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md and https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node. onnx has some tools and guidelines on how to test a new backend implementation.

# Models
In `models` folder, you will find a bunch of models that can be used for testing or debugging. Note that there are also some python scripts to generate that models. The idea here is to generate several types of models, that represent a wide variety of machine learning algorithms, and verify the C output against the Python one.

# Ideas/Help needed
* x

# TODO
*

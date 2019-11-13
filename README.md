# Embedded Machine Learning C ONNX Runtime
Note that this project is in a very early stage so its not production ready yet. Developers are needed so feel free to contact or contribute with a pull request. Use at your own risk. In short, our purpose is to create a pure C runtime for `onnx` with the lowest possible footprint, aimed to small embedded devices.

`embeddedml` provides a `C` runtime that uses `onnx` (Open Neural Network Exchange) models. With this library you can run machine learning inference in pure `C` code, that can compile with `C89` standard. In other words, given an input and a previously trained model, this tool will allow you to predict the output. This project is aimed to fill the gap between `onnx` and small embedded devices, that don't have many resources and where is not possible to use modern compilers. Just train your model with your favourite tool (sk-learn, keras, tf, pytorch), export the `model.onnx` with onnx and use it to feed `embeddedml`. As simple as that. Since `onnx` provides a big set of operators, not all of them will be covered in the first releases. On top of that, the idea of `embeddedml` design is that only the needed operators are compiled into the binaries that you deploy to your device. A full version will be of course also suported.

# High level requirements

* A pure C runtime for onnx shall be developed
* No external libraries shall be used
* Code shall compile with `C99` standard
* Compiled code shall be as little as possible (i.e. fit into a small device)

# Current limitations

* Very few basic operators are implemented.
* Onnx supports many types (`fixed-point`, `float`, `int`,...). Few of them are implemented.
* `has_raw_data` is not supported. A `TensorProto` is assumed to have the data inside any of the structs (int, float,...) and not in raw_data.

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
`cunit` is used to test the code. Two different test levels are written, on operator level (i.e. matrix multiplication) and on a model level (whole model end to end)

* Have a look to https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md and https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node. onnx has some tools and guidelines on how to test a new backend implementation.

# Milestones

- [x] Implement matmul and add operators
- [x] Integrate onnx backend testing
- [ ] Implement all operators contained in MNIST model
- [ ] :  Conv
- [x] :  Add
- [x] :  Relu
- [ ] :  MaxPool
- [ ] :  Reshape
- [x] :  Matmul
- [ ] Run end to end tests for MNIST model
- [ ] Implement a significant amount of onnx operators, most common ones
- [ ] Compile and deploy a model such MNIST into a real embedded device

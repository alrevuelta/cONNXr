# onnx-c runtime
This repo contains a pure C99 runtime to run inference on `onnx` models. You can train your model with you favourite framework (tensorflow, keras, sk-learn, you name it!) and once trained export it to a `.onnx` file, that will be used to run inference. This makes this library totally framework agnostic, no matter how you train your model, this repo will run it using the common interface that `onnx` provides. This runtime was thought for embedded devices, that have low resources and that might not be able to compile newer cpp versions, so the idea is to keep the dependancies as minimum as possible, or even zero. No GPUs or fancy processor architectures, just pure non multi-thread C99 code, compatible with almost any embedded device. Lets allow our IoT devices to run inference on the edge, but without sacrificing the tools that the big AI fishes in the industry provide.
![diag1](https://github.com/alrevuelta/embedded-ml/blob/master/doc/img/diag1.png)

Note that this project is in a very early stage so its not production ready yet. Developers are needed so feel free to contact or contribute with a pull request. Use at your own risk. In short, our purpose is to create a pure C runtime for `onnx` with the lowest possible footprint, aimed to small embedded devices.

# High level requirements

* A pure C runtime for onnx shall be developed
* No external libraries shall be used
* Code shall compile with `C99` standard
* Compiled code shall be as little as possible (i.e. fit into a small device)

# Current limitations

* Very few basic operators are implemented.
* Each operator works with many data types. Only few of them are implemented.
* `has_raw_data` is not supported. A `TensorProto` is assumed to have the data inside any of the structs (int, float,...) and not in raw_data.
* So far memory management is a mess, so you will find a memory leak for sure.

# Run

You will find a simple `Makefile` inside `src`. This has been tested only in Mac so far. Not that you have to link against `-lprotobuf-c` library, that takes care of reading the `protocol-buffers` that are used to store the `onnx` file.

```
cd src
make main
```

# Protocol Buffers
`onnx` uses protocol buffers to serialize the models data. Note that `protobuf-c` is used to generate the `pb/onnx.pb-c.c` and `pb/onnx.pb-c.h`. Files are already provided, but you can generate it like this:

```
protoc --c_out=. onnx.proto
```

In the future `nanopb` might be used, since it can generate smaller files. Investigate also how to use `.option` file. You can find some initial tests in `pb/nanopb` but is not yet being used. You can regenerate it using the following command, but note that you need to have a `protoc` binary.

```
generator-bin/protoc --nanopb_out=. onnx.proto
```

Note that nanopb archieves a signifiant reduction of the '.c' and `.h` files. 69K/45K for non nanopb and 14/20KB for nanopb. So (69+45)/(14+20) thats 3 times less!

# Tests
`cunit` is used to test the code. Two different test levels are written, on operator level (i.e. matrix multiplication) and on a model level (whole model end to end)

You can run all the tests with
```
make test
```

If you are implementing a new operator or debugging, you might want to run only one specific test. The following command will run a specific testcase from its test suite:
```
make test ts=Operators_TestSuite tc=test_operator_maxpool_1d_default
```

* Have a look to https://github.com/onnx/onnx/blob/master/docs/OnnxBackendTest.md and https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node. onnx has some tools and guidelines on how to test a new backend implementation.

# MNIST
// TODO

# Milestones

- [x] Implement matmul and add operators
- [x] Integrate onnx backend testing
- [ ] Implement all operators contained in MNIST model
- [x] Conv
- [x] Add
- [x] Relu
- [x] MaxPool
- [x] Reshape
- [x] Matmul
- [ ] Run end to end tests for MNIST model
- [ ] Implement a significant amount of onnx operators, most common ones
- [ ] Compile and deploy a model such MNIST into a real embedded device

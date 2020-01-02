# onnx-c runtime
![](https://github.com/alrevuelta/embedded-ml/workflows/CI/badge.svg)

> A `onnx` runtime written in pure `C99` with zero dependancies focused on small embedded devices. Run inference on your machine learning models no matter which framework you train it with and no matter the device that you use.

<h2 align="center">📗 Documentation 📗</h2>

<p align="center">
  <a href="doc/01_Introduction.md">01 Introduction</a> •
  <a href="doc/02_CodeOverview.md">02 Code Overview</a> •
  <a href="doc/03_Testing.md">03 Testing</a> •
  <a href="doc/04_Contributing.md">04 Contributing</a> •
  <a href="doc/05_Requirements.md">05 Requirements</a>
</p>

This repo contains a pure C99 runtime to run inference on `onnx` models. You can train your model with you favourite framework (tensorflow, keras, sk-learn, you name it!) and once trained export it to a `.onnx` file, that will be used to run inference. This makes this library totally framework agnostic, no matter how you train your model, this repo will run it using the common interface that `onnx` provides. This runtime was thought for embedded devices, that have low resources and that might not be able to compile newer cpp versions, so the idea is to keep the dependancies as minimum as possible, or even zero. No GPUs or fancy processor architectures, just pure non multi-thread C99 code, compatible with almost any embedded device. Lets allow our IoT devices to run inference on the edge, but without sacrificing the tools that the big AI fishes in the industry provide.

Note that this project is in a very early stage so its not even close to be production ready. Developers are needed so feel free to contact or contribute with a pull request. See **Help Needed** and [doc](doc) for more information about how to contribute. So far we can run inference on the `MNIST` model to recognise handwritten digits.

# Related Projects
Other C/C++ related projects

| Project       | Framework     | Language  | Size |
| ------------- |:-------------:| -----:| ----:|
| [onnxruntime](https://github.com/microsoft/onnxruntime)   | ONNX       | x | x |
| [darknet](https://github.com/pjreddie/darknet)            | ?          | x | x |
| [uTensor](https://github.com/uTensor/uTensor)             | TensorFlow | x | x |
| [nnom](https://github.com/majianjia/nnom)                 | Keras      | x | x |
| [ELL](https://github.com/Microsoft/ELL)                   | ELL        | x | x |


# Install
Check the `Makefile` inside `test` that compiles the code and run a bunch of test cases for the implemented operators + MNIST digit recognition model. Library compilation into a static library is not done yet.

# Example

## In your code
Note that this example won't work as it is. Some more work is needed.

```c
int main()
{
  /* Open the onnx model you want to use*/
  Onnx__ModelProto *model = openOnnxFile("model.onnx");

  /* Populate and alloc memory for your inputs array */
  Onnx__TensorProto **inputs;

  /* Define the number of inputs you have set*/
  int numOfInputs = 1;

  /* Run inference on the model with your inputs*/
  Onnx__TensorProto **output = inference(model, inputs, numOfInputs);

  /* In output you will find an array of tensors with the outputs of each node */

  /* Free all resources */

  return 0;
}
```

## Command Line Interface
A simple command line interface is provided so you can easily use it from your terminal. Note that its still in a very early stage.

Just compile it
```
make build_cli
```

And use it. First parameter is the model, and second the input in `.pb` format. In the future it might support other input formats such as images.
```
./connxr test/mnist/model.onnx test/mnist/test_data_set_0/input_0.pb
```

# Limitations

* Very few basic operators are implemented, so a model that contains a not implemented operator will fail. See them inside `operators` folder
* The only end to end tested model so far is the MNIST one, for handwritten recognition digits.
* Each operator works with many data types (double, float, int16, int32). Only few of them are implemented.
* `has_raw_data` is not supported. A `TensorProto` is assumed to have the data inside any of the structs (int, float,...) and not in raw_data.
* So far memory management is a mess, so you will find a memory leak for sure.
* There are some hardcodings, here and there.

# Help Needed

- [x] Integrate onnx backend testing
- [x] Implement all operators contained in MNIST model
- [x] Run end to end tests for MNIST model
- [ ] Implement a significant amount of onnx operators, most common ones
- [ ] Compile and deploy a model (i.e. MNIST) into a real embedded device
- [x] Set up a nice CI with Azure or GitHub Actions
- [ ] Run profiling on the operators
- [ ] Migrate to nanopb to reduce the size of the pb files
- [ ] Run memory check and leak detection (Valgrind?)
- [ ] Add more tests than the onnx backend, which is not sufficient
- [ ] Create a nice Makefile, compile library as a static library to be linked
- [ ] Try different compilers
- [ ] Enable gcc extra options (pedantic, all W, etc,...)

# cONNXr C ONNX Runtime
![macos-latest](https://github.com/alrevuelta/cONNXr/workflows/macos-latest/badge.svg) ![ubuntu-latest](https://github.com/alrevuelta/cONNXr/workflows/ubuntu-latest/badge.svg) ![windows-latest](https://github.com/alrevuelta/cONNXr/workflows/windows-latest/badge.svg)


> A `onnx` runtime written in pure `C99` with zero dependancies focused on embedded devices. Run inference on your machine learning models no matter which framework you train it with and no matter the device that you use. This is the perfect way to go in old hardware that doesn't support fancy modern C or C++.

<h2 align="center">📗 Documentation 📗</h2>

<p align="center">
  <a href="doc/01_Introduction.md">01 Introduction</a> •
  <a href="doc/02_CodeOverview.md">02 Code Overview</a> •
  <a href="doc/03_Testing.md">03 Testing</a> •
  <a href="doc/04_Contributing.md">04 Contributing</a> •
  <a href="doc/05_Requirements.md">05 Requirements</a>
  <a href="doc/06_OperatorStatus.md">06 Operator Status</a>
</p>

This repo contains a pure C99 runtime to run inference on `onnx` models. You can train your model with you favourite framework (tensorflow, keras, sk-learn, you name it!) and once trained export it to a `.onnx` file, that will be used to run inference. This makes this library totally framework agnostic, no matter how you train your model, this repo will run it using the common interface that `onnx` provides. This runtime was thought for embedded devices, that have low resources and that might not be able to compile newer cpp versions, so the idea is to keep the dependancies as minimum as possible, or even zero. No GPUs or fancy processor architectures, just pure non multi-thread C99 code, compatible with almost any embedded device. Lets allow our IoT devices to run inference on the edge, but without sacrificing the tools that the big AI fishes in the industry provide. Dealing with old hardware? This might be also for you.

Note that this project is in a very early stage so its not even close to be production ready. Developers are needed so feel free to contact or contribute with a pull request. See **Help Needed** and [doc](doc) for more information about how to contribute.

# Out of the box examples

Some very well known models are supported out of the box, just compile the command line as follows and call it with two parameters (first the `ONNX` model, and second the `input` to run inference on). Note that the input has to be a `.pb` file. If you have your own model and its not working, its probably because its using an operator that we haven't implemented yet, so feel free to open an issue and we will happy to help.
```
make build_cli
```

## [MNIST](https://github.com/onnx/models/tree/master/vision/classification/mnist)
```
./connxr test/mnist/model.onnx test/mnist/test_data_set_0/input_0.pb
```

## [tiny YOLO v2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2)
```
./connxr test/tiny_yolov2/Model.onnx test/tiny_yolov2/test_data_set_0/input_0.pb
```

## [super resolution](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016)
```
TODO
```

TODO:
* tiny YOLO v3: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov3 TODO!
* https://lutzroeder.github.io/netron/ TODO
* Quantized MNIST. TODO. Using ONNX MNIST as baseline and quantizing it. Work ongoing

# In you code

If you want to use `cONNXr` as part of your code, you can either include all the files in your project and compile them, or perhaps link it as a static library, but this second option is not supported yet. You can do something like this, but note that this example won't work as it is. Some more work is needed.

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

# Related Projects
Other C/C++ related projects

| Project       | Framework     | Language  | Size |
| ------------- |:-------------:| -----:| ----:|
| [onnxruntime](https://github.com/microsoft/onnxruntime)   | ONNX       | x | x |
| [darknet](https://github.com/pjreddie/darknet)            | ?          | x | x |
| [uTensor](https://github.com/uTensor/uTensor)             | TensorFlow | x | x |
| [nnom](https://github.com/majianjia/nnom)                 | Keras      | x | x |
| [ELL](https://github.com/Microsoft/ELL)                   | ELL        | x | x |
| [TF Lite](xx)                                             | TF Lite    | x | x |
| [plaidML](https://github.com/plaidml/plaidml)             | plaidML    | x | x |

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
- [ ] Implement some "Int" operators and fixed point stuff.
- [ ] Create and run a quantized variation of the MNIST model

# Disclaimer
This project is not associated in any way with ONNX and it is not an official solution nor officially supported by ONNX, it is just an application build on top of the `.onnx` format that aims to help people that want to run inference in devices that are not supported by the official runtimes. Use at your own risk.

# License
TODO

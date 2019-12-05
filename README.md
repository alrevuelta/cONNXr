# onnx-c runtime
![](https://github.com/alrevuelta/embedded-ml/workflows/CI/badge.svg)

This repo contains a pure C99 runtime to run inference on `onnx` models. You can train your model with you favourite framework (tensorflow, keras, sk-learn, you name it!) and once trained export it to a `.onnx` file, that will be used to run inference. This makes this library totally framework agnostic, no matter how you train your model, this repo will run it using the common interface that `onnx` provides. This runtime was thought for embedded devices, that have low resources and that might not be able to compile newer cpp versions, so the idea is to keep the dependancies as minimum as possible, or even zero. No GPUs or fancy processor architectures, just pure non multi-thread C99 code, compatible with almost any embedded device. Lets allow our IoT devices to run inference on the edge, but without sacrificing the tools that the big AI fishes in the industry provide.

<h2 align="center">📗 Index 📗</h2>

<p align="center">
  <a href="doc/01_Introduction.md">01 Introduction</a> •
  <a href="doc/02_CodeOverview.md">02 Code Overview</a> •
  <a href="doc/03_Testing">03 Testing</a> •
  <a href="doc/04_Contributing">04 Contributing</a> •
  <a href="doc/05_Requirements">05 Requirements</a>
</p>

![diag1](https://github.com/alrevuelta/embedded-ml/blob/master/doc/img/diag1.png)

Note that this project is in a very early stage so its not production ready yet. Developers are needed so feel free to contact or contribute with a pull request. Use at your own risk. In short, our purpose is to create a pure C runtime for `onnx` with the lowest possible footprint, aimed to small embedded devices.


# Limitations and Help Needed

* Very few basic operators are implemented.
* Each operator works with many data types. Only few of them are implemented.
* `has_raw_data` is not supported. A `TensorProto` is assumed to have the data inside any of the structs (int, float,...) and not in raw_data.
* So far memory management is a mess, so you will find a memory leak for sure.
* MNIST model is implemented

# Example
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

# Milestones

- [x] Implement matmul and add operators
- [x] Integrate onnx backend testing
- [x] Implement all operators contained in MNIST model
- [x] Conv
- [x] Add
- [x] Relu
- [x] MaxPool
- [x] Reshape
- [x] Matmul
- [x] Run end to end tests for MNIST model
- [ ] Implement a significant amount of onnx operators, most common ones
- [ ] Compile and deploy a model such MNIST into a real embedded device

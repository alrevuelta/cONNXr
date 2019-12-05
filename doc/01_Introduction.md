# 01 Inntroduction
## What is ONNX?
If you don't know about `onnx` you might want to read about it before. They have a nice [website](https://onnx.ai/) and great repositories with a lot of documentation to read about. Everything is open source, and really big companies in the industry are behind it (AMD, ARM, AWS, Nvidia, Ibm) just to name a few.
* https://github.com/onnx/onnx
* https://github.com/onnx/onnx-r
* https://github.com/onnx/models
* https://github.com/owulveryck/onnx-go
* https://github.com/microsoft/onnxruntime

In short, `onnx` provides a **O**pen **N**eural **N**etwork **E**xchange format. This format, describes a huge set of operators, that can be mixed to create every type of machine learning model that you ever heard of, from a simple neural network to complex deep convolutional networks. Some examples of operators are: matrix multiplications, convolutions, addings, maxpool, sin, cosine, you name it! They provide a standarized set of operators [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md). So we can say that `onnx` provides a layer of abstraction to ML models, which makes all framework compatible between them. Exporters are provided for a huge variety of frameworks (PyTorch, TensorFlow, Keras, Scikit-Learn) so if you want to convert a model from Keras to TensorFlow, you just have to use Keras exporter to export `Keras->ONNX` and then use the importer to import `ONNX-TensorFlow`.

In the following image, you can find an example on how a `onnx` model looks like. Its just a bunch of `nodes` that are connected between them to form a `graph`. Each node has an `operator` that takes some `inputs` with some specific `dimensions` and some `attributes` and calculates some `outputs`. This is how the inference is calculated, just forward propagating the input along every node ultil the last one is reached.

![mnist](/doc/img/mnist_example.png)

## What is onnx-c runtime
Well, now that you know about `onnx`, our project is just a runtime that runs inference on `onnx` models, that can be trained with whatever framework you want. The only difference between this runtime and others, is that this one is written in pure `C99` without any dependancy. This means that it should be able to compile with almost any compiler, no matter how old it is. Our goal is to enable small embedded devices that doesn't have much resources or fancy features (like GPUs or any type of hardware accelerator) to run inference. No GPUs, no multithreading, no dependancies, just pure C code with the lowest possible footprint. Train your model in whatever ML framework you want, export it to `.onnx` and deploy it wherever you want.

## Other runtimes
// TODO: Compare this C runtime with other ones (the official onnx one, onnx-r or onnx-go)

## Other non-onnx embedded runtimes
// TODO: Name other non-onnx C runtimes and list advantages and disadvantages.

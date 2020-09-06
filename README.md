# 🤖 cONNXr C ONNX Runtime
![macos-latest](https://github.com/alrevuelta/cONNXr/workflows/macos-latest/badge.svg) ![ubuntu-latest](https://github.com/alrevuelta/cONNXr/workflows/ubuntu-latest/badge.svg) ![windows-latest](https://github.com/alrevuelta/cONNXr/workflows/windows-latest/badge.svg)


> A `onnx` runtime written in pure `C99` with zero dependencies focused on embedded devices. Run inference on your machine learning models no matter which framework you train it with and no matter the device that you use. This is the perfect way to go in old hardware that doesn't support fancy modern C or C++.

# 📗 Documentation

Documentation about the project, how to collaborate, architecture and much more. Available [here](https://connxr.readthedocs.io/)

# 🎓 Introduction

This repo contains a pure C99 runtime to run inference on `onnx` models. You can train your model with you favourite framework (tensorflow, keras, sklearn) and once trained export it to a `.onnx` file, that will be used to run inference. This makes this library totally framework agnostic, no matter how you train your model, this repo will run it using the common interface that `onnx` provides. This runtime was thought for embedded devices, that might not be able to compile newer cpp versions. No GPUs nor HW accelerators, just pure non multi-thread C99 code, compatible with almost any embedded device. Dealing with old hardware? This might be also for you.

This project can be also useful if you are working with some bare metal hardware with dedicated accelerators. If this is the case, you might find useful to reuse the architecture and replace the specific operators by your own ones.

Note that this project is in a very early stage so its not even close to be production ready. Developers are needed so feel free to contact or contribute with a pull request. You can also have a look to the opened issues if you want to contribute, specially the ones labeled for beginners. See contributing section.

# 🖥 Out of the box examples

Some very well known models are supported out of the box, just compile the command line as follows and call it with two parameters (first the `ONNX` model, and second the `input` to run inference on). Note that the input has to be a `.pb` file. If you have your own model and its not working, its probably because its using an operator that we haven't implemented yet, so feel free to open an issue and we will happy to help.
```
make all
```

## [MNIST](https://github.com/onnx/models/tree/master/vision/classification/mnist)
```
build/connxr test/mnist/model.onnx test/mnist/test_data_set_0/input_0.pb
```

## [tiny YOLO v2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2)
```
build/connxr test/tiny_yolov2/Model.onnx test/tiny_yolov2/test_data_set_0/input_0.pb
```

## [super resolution](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016)
```
build/connxr test/super_resolution/super_resolution.onnx test/super_resolution/test_data_set_0/input_0.pb
```

## [mobilenet v2](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)
```
build/connxr test/mobilenetv2-1.0/mobilenetv2-1.0.onnx test/mobilenetv2-1.0/test_data_set_0/input_0.pb
```

TODO:
* [Fast Neural Style Transfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style)
* [TinyTOLOv3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3)
* [Interception_V1](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1)
* [Quantized MNIST](https://github.com/alrevuelta/cONNXr/blob/master/scripts/quantized_model.onnx)

# ⚙ Example

If you want to use `cONNXr` as part of your code, you can either include all the files in your project and compile them, or perhaps link it as a static library, but this second option is not supported yet.

```c
int main()
{
  /* Open your onnx model */
  Onnx__ModelProto *model = openOnnxFile("model.onnx");

  /* Create your input tensor or load a protocol buffer one */
  Onnx__TensorProto *inp0 = openTensorProtoFile("input0.pb");

  /* Set the input name */
  inp0set0->name = model->graph->input[0]->name;

  /* Create the array of inputs to the model */
  Onnx__TensorProto *inputs[] = { inp0set0 };

  /* Resolve all inputs and operators */
  resolve(model, inputs, 1);

  /* Run inference on your input */
  Onnx__TensorProto **output = inference(model, inputs, 1);

  /* Print the last output which is the model output */
  for (int i = 0; i < all_context[_populatedIdx].outputs[0]->n_float_data; i++){
      printf("n_float_data[%d] = %f\n", i, all_context[_populatedIdx].outputs[0]->float_data[i]);
  }
}
```

# 🏷 Related Projects

Other C/C++ related projects: [onnxruntime](https://github.com/microsoft/onnxruntime), [darknet](https://github.com/pjreddie/darknet), [uTensor](https://github.com/uTensor/uTensor), [nnom](https://github.com/majianjia/nnom), [ELL](https://github.com/Microsoft/ELL), [plaidML](https://github.com/plaidml/plaidml), [deepC](https://github.com/ai-techsystems/deepC), [onnc](https://github.com/ONNC/onnc)


# ⛓ Limitations

* Few basic operators are implemented, so a model that contains a not implemented operator will fail.
* Each operator works with many data types (double, float, int16, int32). Only few of them are implemented.
* The reference implementation is with `float`, so you might run into troubles with other types.
* As a general note, this project is a proof of concept/prototype, so bear that in mind.

# 📌 Disclaimer
This project is not associated in any way with ONNX and it is not an official solution nor officially supported by ONNX, it is just an application build on top of the `.onnx` format that aims to help people that want to run inference in devices that are not supported by the official runtimes. Use at your own risk.

# 📗 License
[MIT License](https://github.com/alrevuelta/cONNXr/blob/master/LICENSE)

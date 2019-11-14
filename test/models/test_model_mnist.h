#ifndef TEST_MODEL_MNIST_H
#define TEST_MODEL_MNIST_H

#include "common_models.h"

void test_model_mnist(void)
{
  // Use mnist model to test
  // test_data_set_0
  // test_data_set_1
  // test_data_set_2

  DEBUG_PRINT("Start: test_model_mnist");

  Onnx__ModelProto *model = openOnnxFile("../test/mnist/model.onnx");
  /*Onnx__TensorProto *inp0set0 = openTensorProtoFile("../test/mnist/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0set0 = openTensorProtoFile("../test/mnist/test_data_set_0/output_0.pb");*/

  Debug_PrintModelInformation(model);
  /*
  convertRawDataOfTensorProto(inp0set0);
  convertRawDataOfTensorProto(out0set0);

  Onnx__TensorProto *inputs[] = { inp0set0 };
  Onnx__TensorProto **output = inference(model, inputs, 1);*/

  // TODO Not finished

  DEBUG_PRINT("End: test_model_mnist");
}

#endif

#ifndef TEST_OPERATOR_ARGMAX_H
#define TEST_OPERATOR_ARGMAX_H
#include "common_operators.h"
#include "../../src/operators/argmax.h"

void test_operator_argmax_custom(void)
{
}

// TODO Argmax has lots of tests, only this one is implemented
void test_operator_argmax_default_axis_example(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_argmax_default_axis_example/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_argmax_default_axis_example/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_argmax_default_axis_example/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *inputs[1] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out0);
}

#endif

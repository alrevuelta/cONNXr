#ifndef TEST_OPERATOR_RELU_H
#define TEST_OPERATOR_RELU_H
#include "common_operators.h"
#include "../../src/operators/relu.h"

void test_operator_relu(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_data_set_0/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_relu/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_relu/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *inputs[1] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out0);
}

#endif

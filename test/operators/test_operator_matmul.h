#ifndef TEST_OPERATOR_MATMUL_H
#define TEST_OPERATOR_MATMUL_H
#include "common_operators.h"
#include "../../src/operators/matmul.h"

// node/test_matmul_2d
// node/test_matmul_3d
// node/test_matmul_4d
// node/test_matmulinteger
void test_operator_matmul_2d(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_matmul_2d/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

#endif

#ifndef TEST_OPERATOR_MATMUL_H
#define TEST_OPERATOR_MATMUL_H
#include "common_operators.h"
#include "../../src/operators/matmul.h"

void test_matmul_2d(void)
{
  // TODO: Run the test without running the whole model. Just feed matmul
  // function with the tensors

  // Split this into multiple files
  // node/test_matmul_2d
  // node/test_matmul_3d
  // node/test_matmul_4d
  // node/test_matmulinteger

  // Test 1: test_matmul_2d
  //Onnx__ModelProto *model = openOnnxFile("../test/node/test_matmul_2d/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_matmul(inp0, inp1, result);

  compareAlmostEqualTensorProto(result, out1);

  // Free resources
  /*
  onnx__tensor_proto__free_unpacked(inp0, NULL);
  onnx__tensor_proto__free_unpacked(inp1, NULL);
  onnx__tensor_proto__free_unpacked(out1, NULL);
  onnx__model_proto__free_unpacked(model, NULL);*/
}

#endif

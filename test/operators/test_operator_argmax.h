#ifndef TEST_OPERATOR_ARGMAX_H
#define TEST_OPERATOR_ARGMAX_H
#include "common_operators.h"
#include "../../src/operators/argmax.h"

void test_operator_argmax_custom(void)
{/*
  // 3x2
  float x[] = {-100, 0.1f, 3.0f, 1200.4f, 0, -3.0f};
  int argmax[3];
  int expected[] = {1, 1, 0};
  Operators_ArgMax(x, 3, 2, 1, 0, argmax);

  for (int i = 0; i < 3; i++) {
    CU_ASSERT(argmax[i] == expected[i]);
  }
*/
  // TODO Test with more than 2D, keepaxis and axis
}

// TODO Argmax has lots of tests, only this one is implemented
void test_operator_argmax_default_axis_example(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_argmax_default_axis_example/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_argmax_default_axis_example/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operators_argmax(inp0, 0, 0, result);

  compareAlmostEqualTensorProto(result, out0);
}

#endif

#ifndef TEST_OPERATOR_ADD_H
#define TEST_OPERATOR_ADD_H
#include "common_operators.h"
#include "../../src/operators/add.h"

void test_operator_add_custom1(void)
{
  /*
  float a[] = {1, 2, 3, 4, 5, 6, 7};
  float b[] = {1.1f, 1.2f, 7.3f, 7, 3, 6, 1.9f};
  float expected[] = {2.1f, 3.2f, 10.3f, 11, 8, 12, 8.9f};
  Operators_Add(a, b, 7, ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT);

  for (int i = 0; i < 7; i++) {
    CU_ASSERT(a[i] == expected[i]);
  }
  */
}

void test_operator_add(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_add/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_add/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_add/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_add(inp0, inp1, result);

  compareAlmostEqualTensorProto(result, out0);

}

void test_operator_add_bcast(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_add(inp0, inp1, result);

  compareAlmostEqualTensorProto(result, out0);
}

#endif

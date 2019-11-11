#ifndef TEST_OPERATOR_CAST_H
#define TEST_OPERATOR_CAST_H
#include "common_operators.h"
#include "../../src/operators/cast.h"

void test_operator_cast_FLOAT_to_DOUBLE(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_cast_FLOAT_to_DOUBLE/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_cast_FLOAT_to_DOUBLE/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operators_cast(inp0, result, ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE);

  compareAlmostEqualTensorProto(result, out0);
}

#endif

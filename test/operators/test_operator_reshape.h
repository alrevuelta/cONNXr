#ifndef TEST_OPERATOR_RESHAPE_H
#define TEST_OPERATOR_RESHAPE_H
#include "common_operators.h"
#include "../../src/operators/reshape.h"

void test_operator_reshape_extended_dims(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_extended_dims/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_extended_dims/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_extended_dims/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

// TODO Check inp1 in all the following tests
void test_operator_reshape_negative_dim(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_negative_dim/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_negative_dim/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_negative_dim/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_negative_extended_dims(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_negative_extended_dims/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_negative_extended_dims/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_negative_extended_dims/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_one_dim(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_one_dim/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_one_dim/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_one_dim/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_reduced_dims(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_reduced_dims/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_reduced_dims/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_reduced_dims/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_reordered_all_dims(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_reordered_all_dims/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_reordered_all_dims/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_reordered_all_dims/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_reordered_last_dims(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_reordered_last_dims/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_reordered_last_dims/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_reordered_last_dims/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_zero_and_negative_dim(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_zero_and_negative_dim/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_zero_and_negative_dim/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_zero_and_negative_dim/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

void test_operator_reshape_zero_dim(void)
{
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_reshape_zero_dim/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_reshape_zero_dim/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_reshape_zero_dim/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *result = malloc (sizeof(*result));
  operator_reshape(inp0, inp1, result);
  compareAlmostEqualTensorProto(result, out0);
}

#endif

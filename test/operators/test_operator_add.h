#ifndef TEST_OPERATOR_ADD_H
#define TEST_OPERATOR_ADD_H
#include "common_operators.h"
#include "../../src/operators/add.h"

void test_operator_add_custom1(void)
{
}

void test_operator_add(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_add/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_add/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_add/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_add/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  compareAlmostEqualTensorProto(outputs[0], out0);
}

void test_operator_add_bcast(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_add_bcast/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out0 = openTensorProtoFile("../test/node/test_add_bcast/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out0);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  compareAlmostEqualTensorProto(outputs[0], out0);
}

#endif

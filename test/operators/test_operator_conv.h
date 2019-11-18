#ifndef TEST_OPERATOR_CONV_H
#define TEST_OPERATOR_CONV_H
#include "common_operators.h"
#include "../../src/operators/conv.h"

//test_conv_with_strides_and_asymmetric_padding
//test_conv_with_strides_no_padding
//test_conv_with_strides_padding

void test_operator_conv_with_strides_and_asymmetric_padding(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_conv_with_strides_and_asymmetric_padding/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_conv_with_strides_and_asymmetric_padding/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_conv_with_strides_and_asymmetric_padding/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_conv_with_strides_and_asymmetric_padding/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  //compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_conv_with_strides_no_padding(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_conv_with_strides_no_padding/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_conv_with_strides_no_padding/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_conv_with_strides_no_padding/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_conv_with_strides_no_padding/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  //compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_conv_with_strides_padding(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_conv_with_strides_padding/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_conv_with_strides_padding/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_conv_with_strides_padding/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_conv_with_strides_padding/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[2] = { inp0, inp1 };
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  //compareAlmostEqualTensorProto(outputs[0], out1);
}

#endif

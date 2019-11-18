#ifndef TEST_OPERATOR_MAXPOOL_H
#define TEST_OPERATOR_MAXPOOL_H
#include "common_operators.h"
#include "../../src/operators/maxpool.h"

// Avaialble tests (14)
/*
test_maxpool_1d_default
test_maxpool_2d_ceil
test_maxpool_2d_default
test_maxpool_2d_dilations
test_maxpool_2d_pads
test_maxpool_2d_precomputed_pads
test_maxpool_2d_precomputed_same_upper
test_maxpool_2d_precomputed_strides
test_maxpool_2d_same_lower
test_maxpool_2d_same_upper
test_maxpool_2d_strides
test_maxpool_3d_default
test_maxpool_with_argmax_2d_precomputed_pads
test_maxpool_with_argmax_2d_precomputed_strides
*/
/* Note that maxpool is not tested as some other operators. Since maxpool needs more
than just the inputs (attributes), the onnx model is used to test.*/

void test_operator_maxpool_1d_default(void)
{

  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_1d_default/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_1d_default/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_1d_default/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Debug_PrintModelInformation(model);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);

  // this variable is 0?
  printf("num outputs = %d\n", _outputIdx);
}

void test_operator_maxpool_2d_ceil(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_ceil/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_ceil/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_ceil/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_default(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_default/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_default/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_default/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_dilations(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_dilations/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_dilations/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_dilations/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_pads(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_pads/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_pads/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_pads/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_precomputed_pads(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_precomputed_pads/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_pads/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_pads/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_precomputed_same_upper(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_precomputed_same_upper/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_same_upper/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_same_upper/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}


void test_operator_maxpool_2d_precomputed_strides(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_precomputed_strides/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_strides/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_precomputed_strides/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_same_lower(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_same_lower/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_same_lower/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_same_lower/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_same_upper(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_same_upper/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_same_upper/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_same_upper/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_2d_strides(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_2d_strides/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_2d_strides/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_2d_strides/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_3d_default(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_3d_default/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_3d_default/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_3d_default/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_with_argmax_2d_precomputed_pads(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_with_argmax_2d_precomputed_pads/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_with_argmax_2d_precomputed_pads/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_with_argmax_2d_precomputed_pads/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

void test_operator_maxpool_with_argmax_2d_precomputed_strides(void)
{
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_maxpool_with_argmax_2d_precomputed_strides/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_maxpool_with_argmax_2d_precomputed_strides/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_maxpool_with_argmax_2d_precomputed_strides/test_data_set_0/output_0.pb");

  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = { inp0 };
  Onnx__TensorProto **outputs = inference(model, inputs, 1);

  compareAlmostEqualTensorProto(outputs[0], out1);
}

#endif

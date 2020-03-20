#ifndef TEST_MODEL_SUPER_RESOLUTION_H
#define TEST_MODEL_SUPER_RESOLUTION_H

#include "common_models.h"

void test_model_super_resolution(void)
{
  TRACE_LEVEL0("Start: test_model_super_resolution");

  Onnx__ModelProto *model = openOnnxFile("test/super_resolution/super_resolution.onnx");
  Onnx__TensorProto *inp0set0 = openTensorProtoFile("test/super_resolution/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0set0 = openTensorProtoFile("test/super_resolution/test_data_set_0/output_0.pb");

  debug_prettyprint_model(model);
  convertRawDataOfTensorProto(inp0set0);
  convertRawDataOfTensorProto(out0set0);

  inp0set0->name = model->graph->input[0]->name;
  printf("%s\n", inp0set0->name);

  Onnx__TensorProto *inputs[] = { inp0set0 };
  debug_prettyprint_tensorproto(inp0set0);
  Onnx__TensorProto **output = inference(model, inputs, 1);

  //printf("Will compare output xx = %s", output[xx]->name);
  //compareAlmostEqualTensorProto(output[xx], out0set0);

  TRACE_LEVEL0("End: test_model_super_resolution");
}

#endif

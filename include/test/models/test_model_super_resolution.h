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

  // TODO Dirty trick. I expected the input name to be included in the
  // input_0, but apparently it is not. Dont know if memory for the name
  // is allocated... but it doesnt crash
  inp0set0->name = "input";
  printf("%s\n\n", inp0set0->name);

  //resolve(model, inputs, 1);
  Onnx__TensorProto *inputs[] = { inp0set0 };
  Onnx__TensorProto **output = inference(model, inputs, 1);

  //printf("Will compare output xx = %s", output[xx]->name);
  //compareAlmostEqualTensorProto(output[xx], out0set0);

  TRACE_LEVEL0("End: test_model_super_resolution");
}

#endif

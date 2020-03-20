#ifndef TEST_MODEL_TINYYOLOV2_H
#define TEST_MODEL_TINYYOLOV2_H

#include "common_models.h"

void test_model_tinyyolov2(void)
{
  // Use mnist model to test
  // test_data_set_0
  // test_data_set_1
  // test_data_set_2

  TRACE_LEVEL0("Start: test_model_tinyyolov2");

  Onnx__ModelProto *model = openOnnxFile("test/tiny_yolov2/Model.onnx");
  Onnx__TensorProto *inp0set0 = openTensorProtoFile("test/tiny_yolov2/test_data_set_0/input_0.pb");
  Onnx__TensorProto *out0set0 = openTensorProtoFile("test/tiny_yolov2/test_data_set_0/output_0.pb");

  debug_prettyprint_model(model);
  convertRawDataOfTensorProto(inp0set0);
  convertRawDataOfTensorProto(out0set0);

  inp0set0->name = model->graph->input[0]->name;
  printf("%s\n", inp0set0->name);

  Onnx__TensorProto *inputs[] = { inp0set0 };

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  Onnx__TensorProto **output = inference(model, inputs, 1);
  end = clock();

  // TODO Is CLOCKS_PER_SEC ok to use?
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[benchmark][tinyyolov2] cycles: %f\n", (double) (end - start));
  printf("[benchmark][tinyyolov2] cpu_time_used: %f\n", cpu_time_used);
  printf("[benchmark][tinyyolov2] CLOCKS_PER_SEC: %d\n", CLOCKS_PER_SEC);

  printf("Will compare output 32 = %s", output[32]->name);
  compareAlmostEqualTensorProto(output[32], out0set0);

  TRACE_LEVEL0("End: test_model_tinyyolov2");
}

#endif

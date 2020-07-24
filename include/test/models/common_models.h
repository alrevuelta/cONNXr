#ifndef TEST_COMMON_MODELS_H
#define TEST_COMMON_MODELS_H
#include "trace.h"
#include "inference.h"
#include "utils.h"
#include <stdio.h>
#include <time.h>

int init_Models_TestSuite(void)
{
  return 0;
}

int clean_Models_TestSuite(void)
{
  return 0;
}

/* For a given onnx model and a set of inputs and expected outputs, runs
inference on that model and checks that the outputs are correct. The model_id
is used for prints and debugging purposes. Its also used by the python script
to check the inference time that it took to run that model.
*/
void test_model(
  char *model_id,
  char *model_path,
  char *io_path,
  int n_inputs,
  int n_outputs
){
  TRACE_LEVEL0("Start testing model: %s\n", model_id);

  // So far only 1 input/output is supported
  if (n_inputs > 1 || (n_outputs > 2)){
    fprintf(stderr, "Only models with one input/output are supported\n");
  }

  Onnx__ModelProto *model = openOnnxFile(model_path);

  char inputPath[200];
  strcpy(inputPath, io_path);
  strcat(inputPath, "/input_0.pb");

  char outputPath[200];
  strcpy(outputPath, io_path);
  strcat(outputPath, "/output_0.pb");

  Onnx__TensorProto *inp0 = openTensorProtoFile(inputPath);
  Onnx__TensorProto *out0 = openTensorProtoFile(outputPath);

  //Debug_PrintModelInformation(model);
  debug_prettyprint_model(model);
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(out0);

  inp0->name = model->graph->input[0]->name;
  printf("%s\n", inp0->name);

  Onnx__TensorProto *inputs[] = { inp0 };
  clock_t start, end;
  double cpu_time_used;

  resolve(model, inputs, 1);
  start = clock();
  inference(model, inputs, 1);
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("[benchmark][%s] cycles: %f\n", model_id, (double) (end - start));
  printf("[benchmark][%s] cpu_time_used: %f\n", model_id, cpu_time_used);
  printf("[benchmark][%s] CLOCKS_PER_SEC: %lld\n", model_id, (long long int)CLOCKS_PER_SEC);

  //Asserts the result using the last calculated output.
  printf("Will compare output %d = %s", _populatedIdx, all_context[_populatedIdx].outputs[0]->name);
  compareAlmostEqualTensorProto(all_context[_populatedIdx].outputs[0], out0);

  TRACE_LEVEL0("Finished testing model: %s\n", model_id);
}

#endif

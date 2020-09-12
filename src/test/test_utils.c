#include <glob.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "test/test_utils.h"
#include "trace.h"
#include "inference.h"
#include "utils.h"
#include "onnx.pb-c.h"
#include "math.h"

int compareAlmostEqualTensorProto(Onnx__TensorProto *a, Onnx__TensorProto *b)
{
  printf("Asserting tensors with name: %s, %s\n", a->name, b->name);

  ASSERT_TRUE(a->data_type == b->data_type);
  printf("data_type: %d,%d ok\n", a->data_type, b->data_type);

  ASSERT_TRUE(a->n_dims == b->n_dims);
  printf("n_dims: %zu,%zu ok\n", a->n_dims, b->n_dims);
   
  for (int d = 0; d < a->n_dims; d++)
  {
    ASSERT_TRUE(a->dims[d] == b->dims[d]);
    printf("dims[%d]: %ld,%ld ok\n", d, a->dims[d], b->dims[d]);
  }

  // TODO Not all types are implemented
  switch(a->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      //CU_FAIL("Data types is undefined");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      ASSERT_TRUE(a->n_float_data == b->n_float_data);
      printf("n_float_data %zu,%zu ok\n", a->n_float_data, b->n_float_data);
      for(int i = 0; i < a->n_float_data; i++)
      {
        if (fabs(a->float_data[i] - b->float_data[i]) > FLOAT_TOLERANCE){
          printf("Does not match %i, %f, %f\n", i, a->float_data[i], b->float_data[i]);
        }
        ASSERT_TRUE(fabs(a->float_data[i] - b->float_data[i]) < FLOAT_TOLERANCE);
      }
      printf("float_data ok\n");
      break;
    /* TODO Merge uint8 with 16 and 32 since the type is the same */
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      printf("ASSERTING EQUAL: %zu, %zu\n", a->n_int32_data, b->n_int32_data);
      ASSERT_TRUE(a->n_int32_data == b->n_int32_data);
      for(int i = 0; i < a->n_int32_data; i++)
      {
        printf("ASSERTING EQUAL: %d, %d\n", a->int32_data[i], b->int32_data[i]);
        ASSERT_TRUE(a->int32_data[i] == b->int32_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      //CU_FAIL("int8 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      //CU_FAIL("uint16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      //CU_FAIL("int16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      printf("ASSERTING EQUAL: %zu, %zu\n", a->n_int32_data, b->n_int32_data);
      ASSERT_TRUE(a->n_int32_data == b->n_int32_data);
      for(int i = 0; i < a->n_int32_data; i++)
      {
        TRACE_LEVEL0("ASSERTING EQUAL: %d, %d\n", a->int32_data[i], b->int32_data[i]);
        ASSERT_TRUE(a->int32_data[i] == b->int32_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      ASSERT_TRUE(a->n_int64_data == b->n_int64_data);
      for(int i = 0; i < a->n_int64_data; i++)
      {
        printf("ASSERTING EQUAL: %" PRId64 ", %" PRId64 "\n", a->int64_data[i], b->int64_data[i]);
        ASSERT_TRUE(a->int64_data[i] == b->int64_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      //CU_FAIL("string data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      //CU_FAIL("bool data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      //CU_FAIL("float16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      ASSERT_TRUE(a->n_double_data == b->n_double_data);
      for(int i = 0; i < a->n_double_data; i++)
      {
        printf("ASSERTING EQUAL: %lf, %lf", a->double_data[i], b->double_data[i]);
        ASSERT_TRUE(a->double_data[i] == b->double_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      //CU_FAIL("uint32 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      //CU_FAIL("uint64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      //CU_FAIL("complex64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      //CU_FAIL("complex128 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      //CU_FAIL("bfloat16 data_type is not implemented");
      break;
    default:
      //CU_FAIL("Unknown data_type");
      break;
  }

  return 0;
}

int test_operator(char *outputName)
{
  /* TODO:
   * - Run across all data_set_xx (only 0 is used)
   * - Only output_0 is assumed. Read N outputs
   */
  printf("\n\nTest %s:\n", outputName);

  char modelPath[200];
  strcpy(modelPath, "test/node/");
  strcat(modelPath, outputName);
  strcat(modelPath, "/");
  strcat(modelPath, "model.onnx");

  printf("Reading model %s ...", modelPath);
  Onnx__ModelProto *model = openOnnxFile(modelPath);
  printf("ok\n");

  char inputPath[200];
  strcpy(inputPath, "test/node/");
  strcat(inputPath, outputName);
  strcat(inputPath, "/test_data_set_0/input_*.pb");

  /* Lazy, just set a huge number of inputs */
  Onnx__TensorProto *inputs[10];
  glob_t globbuf;
  int nInputs = 0;
  if (0==glob(inputPath, 0, NULL, &globbuf)){
    char **inputPbs=globbuf.gl_pathv;
    for(;*inputPbs;inputPbs++){
      printf("Reading input %s ...", *inputPbs);
      Onnx__TensorProto *inputN = openTensorProtoFile(*inputPbs);
      convertRawDataOfTensorProto(inputN);
      inputs[nInputs] = inputN;
      nInputs++;
      printf("ok\n");
    }
  }
  globfree(&globbuf);

  char outputPath[200];
  strcpy(outputPath, "test/node/");
  strcat(outputPath, outputName);
  strcat(outputPath, "/test_data_set_0/output_0.pb");

  printf("Reading output %s ...", outputPath);
  Onnx__TensorProto *out0set0 = openTensorProtoFile(outputPath);
  convertRawDataOfTensorProto(out0set0);
  printf("ok\n");

  resolve(model, inputs, nInputs);
  printf("Running inference\n");
  inference(model, inputs, nInputs);

  /* Some operators have more than two outputs to assert */
  printf("Will compare output %d = %s\n", _populatedIdx, all_context[_populatedIdx].outputs[0]->name);
  int result = compareAlmostEqualTensorProto(all_context[_populatedIdx].outputs[0], out0set0);

  return result;
}

/* For a given onnx model and a set of inputs and expected outputs, runs
inference on that model and checks that the outputs are correct. The model_id
is used for prints and debugging purposes. Its also used by the python script
to check the inference time that it took to run that model.

The output is the inference time. If the test failed, its < 0
*/
double test_model(
  char *model_id,
  char *model_path,
  char *io_path,
  int n_inputs,
  int n_outputs
){
  TRACE_LEVEL0("\n\nStart testing model: %s\n", model_id);

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
  int exit_status = compareAlmostEqualTensorProto(all_context[_populatedIdx].outputs[0], out0);

  TRACE_LEVEL0("Finished testing model: %s\n", model_id);

  return exit_status == 0 ? cpu_time_used: -1;
}
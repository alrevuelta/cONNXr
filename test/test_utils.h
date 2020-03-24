#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <glob.h>

#include "../src/trace.h"
#include "../src/inference.h"
#include "../src/utils.h"
#include <glob.h>

#define FLOAT_TOLERANCE 0.001f

// Compare if equal with some tolarenace
void compareAlmostEqualTensorProto(Onnx__TensorProto *a, Onnx__TensorProto *b)
{
  printf("\nAsserting, a dims:\n");
  debug_print_dims(a->n_dims, a->dims);
  printf("\nAsserting, b dims:\n");
  debug_print_dims(b->n_dims, b->dims);
  printf("\nAsserting, a data_type: %d\n", a->data_type);
  CU_ASSERT_EQUAL(a->data_type, b->data_type);
  printf("\nAsserting, b data_type: %d\n", b->data_type);
  CU_ASSERT_EQUAL(a->n_dims, b->n_dims);
  for (int d = 0; d < a->n_dims; d++)
  {
    CU_ASSERT_EQUAL(a->dims[d], b->dims[d]);
  }

  // TODO Not all types are implemented
  switch(a->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      CU_FAIL("Data types is undefined");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      CU_ASSERT_EQUAL(a->n_float_data, b->n_float_data);
      for(int i = 0; i < a->n_float_data; i++)
      {
        if (fabs(a->float_data[i] - b->float_data[i]) > FLOAT_TOLERANCE){
          TRACE_LEVEL0("%i, %f, %f\n", i, a->float_data[i], b->float_data[i]);
        }
        CU_ASSERT(fabs(a->float_data[i] - b->float_data[i]) < FLOAT_TOLERANCE);
      }
      break;
    /* TODO Merge uint8 with 16 and 32 since the type is the same */
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      TRACE_LEVEL0("ASSERTING EQUAL: %zu, %zu\n", a->n_int32_data, b->n_int32_data);
      CU_ASSERT_EQUAL(a->n_int32_data, b->n_int32_data);
      for(int i = 0; i < a->n_int32_data; i++)
      {
        TRACE_LEVEL0("ASSERTING EQUAL: %d, %d\n", a->int32_data[i], b->int32_data[i]);
        CU_ASSERT_EQUAL(a->int32_data[i], b->int32_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      CU_FAIL("int8 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      CU_FAIL("uint16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      CU_FAIL("int16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      TRACE_LEVEL0("ASSERTING EQUAL: %zu, %zu\n", a->n_int32_data, b->n_int32_data);
      CU_ASSERT_EQUAL(a->n_int32_data, b->n_int32_data);
      for(int i = 0; i < a->n_int32_data; i++)
      {
        TRACE_LEVEL0("ASSERTING EQUAL: %d, %d\n", a->int32_data[i], b->int32_data[i]);
        CU_ASSERT_EQUAL(a->int32_data[i], b->int32_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      CU_ASSERT_EQUAL(a->n_int64_data, b->n_int64_data);
      for(int i = 0; i < a->n_int64_data; i++)
      {
        TRACE_LEVEL0("ASSERTING EQUAL: %ld, %ld\n", a->int64_data[i], b->int64_data[i]);
        CU_ASSERT_EQUAL(a->int64_data[i], b->int64_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      CU_FAIL("string data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      CU_FAIL("bool data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      CU_FAIL("float16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      CU_ASSERT_EQUAL(a->n_double_data, b->n_double_data);
      for(int i = 0; i < a->n_double_data; i++)
      {
        TRACE_LEVEL0("ASSERTING EQUAL: %lf, %lf", a->double_data[i], b->double_data[i]);
        CU_ASSERT_EQUAL(a->double_data[i], b->double_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      CU_FAIL("uint32 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      CU_FAIL("uint64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      CU_FAIL("complex64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      CU_FAIL("complex128 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      CU_FAIL("bfloat16 data_type is not implemented");
      break;
    default:
      CU_FAIL("Unknown data_type");
      break;
  }
}

void testOperator(char *outputName)
{
  /* TODO:
   * - Run across all data_set_xx (only 0 is used)
   * - Only output_0 is assumed. Read N outputs
   */
  char modelPath[200];
  strcpy(modelPath, "test/node/");
  strcat(modelPath, outputName);
  strcat(modelPath, "/");
  strcat(modelPath, "model.onnx");

  printf("Reading model %s\n", modelPath);
  Onnx__ModelProto *model = openOnnxFile(modelPath);

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
      printf("Reading input %s\n", *inputPbs);
      Onnx__TensorProto *inputN = openTensorProtoFile(*inputPbs);
      convertRawDataOfTensorProto(inputN);
      inputs[nInputs] = inputN;
      nInputs++;
    }
  }
  globfree(&globbuf);

  char outputPath[200];
  strcpy(outputPath, "test/node/");
  strcat(outputPath, outputName);
  strcat(outputPath, "/test_data_set_0/output_0.pb");

  printf("Reading output %s\n", outputPath);
  Onnx__TensorProto *out0set0 = openTensorProtoFile(outputPath);
  convertRawDataOfTensorProto(out0set0);

  printf("Running inference\n");
  Onnx__TensorProto **output = inference(model, inputs, nInputs);

  /* Some operators have more than two outputs to assert */
  int outputToAssert = 0;
  TRACE_LEVEL0("Asserting output %s", output[outputToAssert]->name);

  compareAlmostEqualTensorProto(output[outputToAssert], out0set0);
}

#endif

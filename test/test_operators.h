#ifndef TEST_OPERATORS_H
#define TEST_OPERATORS_H

#include "test_utils.h"
#include "../src/embeddedml_debug.h"
#include "../src/embeddedml_inference.h"
#include "../src/embeddedml_utils.h"


int init_Operators_TestSuite(void)
{
  return 0;
}

int clean_Operators_TestSuite(void)
{
  return 0;
}

void test_Operators_MatMul(void)
{
  // node/test_matmul_2d
  // node/test_matmul_3d
  // node/test_matmul_4d
  // node/test_matmulinteger

  // Test 1: test_matmul_2d
  Onnx__ModelProto *model = openOnnxFile("../test/node/test_matmul_2d/model.onnx");
  Onnx__TensorProto *inp0 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_0.pb");
  Onnx__TensorProto *inp1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/input_1.pb");
  Onnx__TensorProto *out1 = openTensorProtoFile("../test/node/test_matmul_2d/test_data_set_0/output_0.pb");

  // Tensors have raw_data. Parse it and store into a normal field for the sake of simplicity
  convertRawDataOfTensorProto(inp0);
  convertRawDataOfTensorProto(inp1);
  convertRawDataOfTensorProto(out1);

  Onnx__TensorProto *inputs[] = {inp0, inp1};
  Onnx__TensorProto **outputs = inference(model, inputs, 2);

  compareAlmostEqualTensorProto(outputs[0], out1);

  // Free resources
  /*
  onnx__tensor_proto__free_unpacked(inp0, NULL);
  onnx__tensor_proto__free_unpacked(inp1, NULL);
  onnx__tensor_proto__free_unpacked(out1, NULL);
  onnx__model_proto__free_unpacked(model, NULL);*/
}

void test_Operators_Add(void)
{
  /*
  float a[] = {1, 2, 3, 4, 5, 6, 7};
  float b[] = {1.1f, 1.2f, 7.3f, 7, 3, 6, 1.9f};
  float expected[] = {2.1f, 3.2f, 10.3f, 11, 8, 12, 8.9f};
  Operators_Add(a, b, 7, ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT);

  for (int i = 0; i < 7; i++) {
    CU_ASSERT(a[i] == expected[i]);
  }
  */
}

void test_Operators_Sigmoid(void)
{
  /*
  float x[] = {-4, -3, -2, -1, 0, 1.1f, 2, 6.7f};
  float expected[] = {0.01798620996, 0.04742587318, 0.119202922, 0.2689414214, 0.5, 0.7502601056, 0.880797078, 0.9987706014};
  Operators_Sigmoid(x, 8);

  for (int i = 0; i < 8; i++) {
    CU_ASSERT(x[i] == expected[i]);
  }*/
}

void test_Operators_Softmax(void)
{/*
  float x[] = {-1, 0, 1};
  float expected[] = {0.09003057, 0.24472847, 0.66524100};
  Operators_Softmax(x, 3, 0);
  for (int i = 0; i < 3; i++) {
    CU_ASSERT(x[i] == expected[i]);
  }*/

  // TODO Implement and test 2 dimensions.
}

void test_Operators_ArgMax(void)
{/*
  // 3x2
  float x[] = {-100, 0.1f, 3.0f, 1200.4f, 0, -3.0f};
  int argmax[3];
  int expected[] = {1, 1, 0};
  Operators_ArgMax(x, 3, 2, 1, 0, argmax);

  for (int i = 0; i < 3; i++) {
    CU_ASSERT(argmax[i] == expected[i]);
  }
*/
  // TODO Test with more than 2D, keepaxis and axis
}

#endif

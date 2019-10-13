#ifndef TEST_OPERATORS_H
#define TEST_OPERATORS_H

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
  // TODO test different sizes
  // TODO test all variants (float, int,...)
  float matrixA[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // 4x3
  float matrixB[]  = {9, 8, 7, 6, 5, 4, 3, 2, 1};              // 3x3
  float matrixC[]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};     // 4x3
  float expected[] = {30, 24, 18, 84, 69, 54, 138, 114, 90, 192, 159, 126};
  Operators_MatMul(matrixA,
                   matrixB,
                   4,
                   3,
                   3,
                   matrixC,
                   ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT);

  // TODO Move this to util function
  for (int i = 0; i < 12; i++)
  {
    CU_ASSERT(matrixC[i]==expected[i]);
  }
}

void test_Operators_Add(void)
{
  float a[] = {1, 2, 3, 4, 5, 6, 7};
  float b[] = {1.1f, 1.2f, 7.3f, 7, 3, 6, 1.9f};
  float expected[] = {2.1f, 3.2f, 10.3f, 11, 8, 12, 8.9f};
  Operators_Add(a, b, 7, ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT);

  for (int i = 0; i < 7; i++) {
    CU_ASSERT(a[i] == expected[i]);
  }
}

void test_Operators_Sigmoid(void)
{
  float x[] = {-4, -3, -2, -1, 0, 1.1f, 2, 6.7f};
  float expected[] = {0.01798620996, 0.04742587318, 0.119202922, 0.2689414214, 0.5, 0.7502601056, 0.880797078, 0.9987706014};
  Operators_Sigmoid(x, 8);

  for (int i = 0; i < 8; i++) {
    CU_ASSERT(x[i] == expected[i]);
  }
}

void test_Operators_Softmax(void)
{
  float x[] = {-1, 0, 1};
  float expected[] = {0.09003057, 0.24472847, 0.66524100};
  Operators_Softmax(x, 3, 0);
  for (int i = 0; i < 3; i++) {
    CU_ASSERT(x[i] == expected[i]);
  }

  // TODO Implement and test 2 dimensions.
}

void test_Operators_ArgMax(void)
{
  // 3x2
  float x[] = {-100, 0.1f, 3.0f, 1200.4f, 0, -3.0f};
  int argmax[3];
  int expected[] = {1, 1, 0};
  Operators_ArgMax(x, 3, 2, 1, 0, argmax);

  for (int i = 0; i < 3; i++) {
    CU_ASSERT(argmax[i] == expected[i]);
  }

  // TODO Test with more than 2D, keepaxis and axis
}

#endif

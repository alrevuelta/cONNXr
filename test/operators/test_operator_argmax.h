#ifndef TEST_OPERATOR_ARGMAX_H
#define TEST_OPERATOR_ARGMAX_H
#include "common_operators.h"

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

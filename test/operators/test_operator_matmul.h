#ifndef TEST_OPERATOR_MATMUL_H
#define TEST_OPERATOR_MATMUL_H
#include "common_operators.h"
#include "../../src/operators/matmul.h"

// node/test_matmul_2d
// node/test_matmul_3d
// node/test_matmul_4d
void test_operator_matmul_2d(void)
{
  testOperator("test_matmul_2d");
}

void test_operator_matmul_3d(void)
{
  testOperator("test_matmul_3d");
}

void test_operator_matmul_4d(void)
{
  testOperator("test_matmul_4d");
}

#endif

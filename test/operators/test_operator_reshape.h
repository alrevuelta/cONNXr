#ifndef TEST_OPERATOR_RESHAPE_H
#define TEST_OPERATOR_RESHAPE_H
#include "common_operators.h"
#include "../../src/operators/reshape.h"

void test_operator_reshape_extended_dims(void)
{
  testOperator("test_reshape_extended_dims");
}

// TODO Check inp1 in all the following tests
void test_operator_reshape_negative_dim(void)
{
  testOperator("test_reshape_negative_dim");
}

void test_operator_reshape_negative_extended_dims(void)
{
  testOperator("test_reshape_negative_extended_dims");
}

void test_operator_reshape_one_dim(void)
{
  testOperator("test_reshape_one_dim");
}

void test_operator_reshape_reduced_dims(void)
{
  testOperator("test_reshape_reduced_dims");
}

void test_operator_reshape_reordered_all_dims(void)
{
  testOperator("test_reshape_reordered_all_dims");
}

void test_operator_reshape_reordered_last_dims(void)
{
  testOperator("test_reshape_reordered_last_dims");
}

void test_operator_reshape_zero_and_negative_dim(void)
{
  testOperator("test_reshape_zero_and_negative_dim");
}

void test_operator_reshape_zero_dim(void)
{
  testOperator("test_reshape_zero_dim");
}

#endif

#ifndef TEST_OPERATOR_MAXPOOL_H
#define TEST_OPERATOR_MAXPOOL_H
#include "common_operators.h"
#include "../../src/operators/maxpool.h"

// Avaialble tests (14)
/*
test_maxpool_1d_default
test_maxpool_2d_ceil
test_maxpool_2d_default
test_maxpool_2d_dilations
test_maxpool_2d_pads
test_maxpool_2d_precomputed_pads
test_maxpool_2d_precomputed_same_upper
test_maxpool_2d_precomputed_strides
test_maxpool_2d_same_lower
test_maxpool_2d_same_upper
test_maxpool_2d_strides
test_maxpool_3d_default
test_maxpool_with_argmax_2d_precomputed_pads
test_maxpool_with_argmax_2d_precomputed_strides
*/
void test_operator_maxpool_1d_default(void)
{
  testOperator("test_maxpool_1d_default");
}

void test_operator_maxpool_2d_ceil(void)
{
  testOperator("test_maxpool_2d_ceil");
}

void test_operator_maxpool_2d_default(void)
{
  testOperator("test_maxpool_2d_default");
}

void test_operator_maxpool_2d_dilations(void)
{
  testOperator("test_maxpool_2d_dilations");
}

void test_operator_maxpool_2d_pads(void)
{
  testOperator("test_maxpool_2d_pads");
}

void test_operator_maxpool_2d_precomputed_pads(void)
{
  testOperator("test_maxpool_2d_precomputed_pads");
}

void test_operator_maxpool_2d_precomputed_same_upper(void)
{
  testOperator("test_maxpool_2d_precomputed_same_upper");
}


void test_operator_maxpool_2d_precomputed_strides(void)
{
  testOperator("test_maxpool_2d_precomputed_strides");
}

void test_operator_maxpool_2d_same_lower(void)
{
  testOperator("test_maxpool_2d_same_lower");
}

void test_operator_maxpool_2d_same_upper(void)
{
  testOperator("test_maxpool_2d_same_upper");
}

void test_operator_maxpool_2d_strides(void)
{
  testOperator("test_maxpool_2d_strides");
}

void test_operator_maxpool_3d_default(void)
{
  testOperator("test_maxpool_3d_default");
}

void test_operator_maxpool_with_argmax_2d_precomputed_pads(void)
{
  testOperator("test_maxpool_with_argmax_2d_precomputed_pads");
}

void test_operator_maxpool_with_argmax_2d_precomputed_strides(void)
{
  testOperator("test_maxpool_with_argmax_2d_precomputed_strides");
}

#endif

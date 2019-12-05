#ifndef TEST_OPERATOR_CONV_H
#define TEST_OPERATOR_CONV_H
#include "common_operators.h"
#include "../../src/operators/conv.h"

//test_conv_with_strides_and_asymmetric_padding
//test_conv_with_strides_no_padding
//test_conv_with_strides_padding

void test_operator_conv_with_strides_and_asymmetric_padding(void)
{
  testOperator("test_conv_with_strides_and_asymmetric_padding");
}

void test_operator_conv_with_strides_no_padding(void)
{
  testOperator("test_conv_with_strides_no_padding");
}

void test_operator_conv_with_strides_padding(void)
{
  testOperator("test_conv_with_strides_padding");
}

#endif

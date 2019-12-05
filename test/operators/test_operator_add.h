#ifndef TEST_OPERATOR_ADD_H
#define TEST_OPERATOR_ADD_H
#include "common_operators.h"
#include "../test_utils.h"
#include "../../src/operators/add.h"

void test_operator_add_custom1(void)
{
}

void test_operator_add(void)
{
  testOperator("test_add");
}

void test_operator_add_bcast(void)
{
  testOperator("test_add_bcast");
}

#endif

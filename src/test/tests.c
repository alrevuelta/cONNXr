#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx.pb-c.h"
#include "trace.h"

void test1(void){
  // Create tests as needed
}

int init_unit_tests(void)
{
  return 0;
}

int clean_unit_tests(void)
{
  return 0;
}

/*
This file is currently unused. All operator and model tests
are now called from Python.
Use this file to write tests that are not related to an operator
or a model. For example, if you want to test a function that you wrote,
this is the place.
*/
int main(int argc, char **argv)
{
  CU_pSuite unit_tests = NULL;

  /* Initialize CUnit test registry */
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  unit_tests = CU_add_suite("unit_tests",
                            init_unit_tests,
                            clean_unit_tests);

  if (NULL == unit_tests) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // Dummy test. Modify as needed.
  CU_add_test(unit_tests, "test1", test1);
  
  CU_basic_set_mode(CU_BRM_VERBOSE);

  CU_basic_run_tests();

  printf("CU_get_number_of_tests_run is %d\n", CU_get_number_of_tests_run());
  printf("CU_get_number_of_tests_failed is %d\n", CU_get_number_of_tests_failed());
  
  if (CU_get_number_of_tests_failed() != 0){
    exit (1);
  }

  CU_cleanup_registry();
  return CU_get_error();
}

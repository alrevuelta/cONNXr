#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <string.h>

#include "../src/embeddedml_opwrapper.h"
#include "../src/embeddedml_operators.h"
#include "../src/onnx.pb-c.h"

// TODO: Future ideas:
// Since many Operators are defined using Python numpy, it would be nice to
// be able to run Python from C to test the operators written in C code. This
// will save a lot of time, and no "expected" vector would be needed.

int init_suite(void)
{
  return 0;
}

int clean_suite(void)
{
  return 0;
}

void test_Operators_MatMul(void)
{
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
  for (int i = 0; i < 12; i++)
  {
    CU_ASSERT(matrixC[i]==expected[i]);
  }
}

void dummy_test2(void)
{
  CU_ASSERT(1==1);
}

int main (void)
{
  CU_pSuite pSuite1,pSuite2 = NULL;

  // Initialize CUnit test registry
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  // Add suite1 to registry
  pSuite1 = CU_add_suite("Basic_Test_Suite1", init_suite, clean_suite);
  if (NULL == pSuite1) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // add test1 to suite1
  if ((NULL == CU_add_test(pSuite1, "test_Operators_MatMul", test_Operators_MatMul)))
  {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // add test 2 to suite2
  if ((NULL == CU_add_test(pSuite1, "dummy_test2", dummy_test2)))
  {
    CU_cleanup_registry();
    return CU_get_error();
  }

  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();

}

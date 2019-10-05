#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <string.h>

int init_suite(void)
{
  return 0;
}

int clean_suite(void)
{
  return 0;
}

void dummy_test1(void)
{
  CU_ASSERT(1==1);
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
  if ((NULL == CU_add_test(pSuite1, "dummy_test1", dummy_test1)))
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

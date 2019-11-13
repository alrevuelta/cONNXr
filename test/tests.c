#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/onnx.pb-c.h"
#include "../src/embeddedml_debug.h"
#include "test_utils.h"

// Operators tests
#include "operators/test_operator_add.h"
#include "operators/test_operator_argmax.h"
#include "operators/test_operator_arrayfeatureextractor.h"
#include "operators/test_operator_cast.h"
#include "operators/test_operator_conv.h"
#include "operators/test_operator_matmul.h"
#include "operators/test_operator_maxpool.h"
#include "operators/test_operator_relu.h"
#include "operators/test_operator_reshape.h"
#include "operators/test_operator_sigmoid.h"
#include "operators/test_operator_softmax.h"
#include "operators/test_operator_zipmap.h"

// Model tests
#include "models/test_model_mnist.h"

// Common stuff
#include "operators/common_operators.h"
#include "models/common_models.h"

int main (void)
{
  CU_pSuite operatorsTestSuite = NULL;
  CU_pSuite modelsTestSuite = NULL;

  // Initialize CUnit test registry
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  // Operators test suite and test cases
  operatorsTestSuite = CU_add_suite("Operators_TestSuite",
                                    init_Operators_TestSuite,
                                    clean_Operators_TestSuite);
  if (NULL == operatorsTestSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // Add tests for Operators test suite
  CU_add_test(operatorsTestSuite, "test_operator_add", test_operator_add);
  CU_add_test(operatorsTestSuite, "test_operator_add_bcast", test_operator_add_bcast);

  CU_add_test(operatorsTestSuite, "test_operator_argmax_default_axis_example", test_operator_argmax_default_axis_example);
  CU_add_test(operatorsTestSuite, "test_operator_arrayfeatureextractor", test_operator_arrayfeatureextractor);
  CU_add_test(operatorsTestSuite, "test_operator_cast_FLOAT_to_DOUBLE", test_operator_cast_FLOAT_to_DOUBLE);
  CU_add_test(operatorsTestSuite, "test_operator_matmul_2d", test_operator_matmul_2d);
  CU_add_test(operatorsTestSuite, "test_operator_relu", test_operator_relu);

  CU_add_test(operatorsTestSuite, "test_operator_reshape_extended_dims", test_operator_reshape_extended_dims);
  /*CU_add_test(operatorsTestSuite, "test_operator_reshape_negative_dim", test_operator_reshape_negative_dim);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_negative_extended_dims", test_operator_reshape_negative_extended_dims);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_one_dim", test_operator_reshape_one_dim);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_reduced_dims", test_operator_reshape_reduced_dims);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_reordered_all_dims", test_operator_reshape_reordered_all_dims);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_reordered_last_dims", test_operator_reshape_reordered_last_dims);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_zero_and_negative_dim", test_operator_reshape_zero_and_negative_dim);
  CU_add_test(operatorsTestSuite, "test_operator_reshape_zero_dim", test_operator_reshape_zero_dim);
  */
  CU_add_test(operatorsTestSuite, "test_operator_sigmoid", test_operator_sigmoid);
  CU_add_test(operatorsTestSuite, "test_operator_softmax", test_operator_softmax);

  // Models test suite and test cases
  modelsTestSuite = CU_add_suite("Models_TestSuite",
                                 init_Models_TestSuite,
                                 clean_Models_TestSuite);
  if (NULL == modelsTestSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  //CU_add_test(modelsTestSuite, "test_model_mnist", test_model_mnist);

  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();

}

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/pb/onnx.pb-c.h"
#include "../src/trace.h"
#include "test_utils.h"

#include "test_onnx_backend_operators.h"
#include "models/test_model_mnist.h"
#include "models/test_model_tinyyolov2.h"
#include "models/test_model_super_resolution.h"
#include "models/common_models.h"

int main(int argc, char **argv)
{
  CU_pSuite onnxBackendSuite = NULL;
  CU_pSuite modelsTestSuite  = NULL;

  /* Initialize CUnit test registry */
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  /* Add onnx test (operators) suite */
  onnxBackendSuite = CU_add_suite("onnxBackendSuite",
                                   init_onnxBackendSuite,
                                   clean_onnxBackendSuite);

  /* Add models test suite*/
  modelsTestSuite = CU_add_suite("modelsTestSuite",
                                 init_Models_TestSuite,
                                 clean_Models_TestSuite);
  if (NULL == onnxBackendSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  if (NULL == modelsTestSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* If arguments are provided, specific test from a suite can be run */
  if (argc == 3)
  {
    printf("Run tc=%s, from suite=%s\n", argv[2], argv[1]);
  }

  /* ONNX Backent Tests */
  /* All the following test cases are part of the official onnx backend
   * tests (see docs/OnnxBackendTest.md at onnx GitHub repo)
   * If a new operator is created, just comment it and a test will run
   * for it. Note also that if the official tests change, this will need to
   * be updated
   */

   //CU_ADD_TEST(onnxBackendSuite, test_abs);
   //CU_ADD_TEST(onnxBackendSuite, test_acos);
   //CU_ADD_TEST(onnxBackendSuite, test_acos_example);
   //CU_ADD_TEST(onnxBackendSuite, test_acosh);
   //CU_ADD_TEST(onnxBackendSuite, test_acosh_example);
   CU_ADD_TEST(onnxBackendSuite, test_add);
   CU_ADD_TEST(onnxBackendSuite, test_add_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_and2d);
   //CU_ADD_TEST(onnxBackendSuite, test_and3d);
   //CU_ADD_TEST(onnxBackendSuite, test_and4d);
   //CU_ADD_TEST(onnxBackendSuite, test_and_bcast3v1d);
   //CU_ADD_TEST(onnxBackendSuite, test_and_bcast3v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_and_bcast4v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_and_bcast4v3d);
   //CU_ADD_TEST(onnxBackendSuite, test_and_bcast4v4d);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_default_axis_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_default_axis_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_default_axis_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_default_axis_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_negative_axis_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_negative_axis_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_negative_axis_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_negative_axis_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_no_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_no_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_no_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmax_no_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_default_axis_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_default_axis_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_default_axis_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_default_axis_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_negative_axis_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_negative_axis_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_negative_axis_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_negative_axis_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_no_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_no_keepdims_example_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_no_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_argmin_no_keepdims_random_select_last_index);
   //CU_ADD_TEST(onnxBackendSuite, test_asin);
   //CU_ADD_TEST(onnxBackendSuite, test_asin_example);
   //CU_ADD_TEST(onnxBackendSuite, test_asinh);
   //CU_ADD_TEST(onnxBackendSuite, test_asinh_example);
   //CU_ADD_TEST(onnxBackendSuite, test_atan);
   //CU_ADD_TEST(onnxBackendSuite, test_atan_example);
   //CU_ADD_TEST(onnxBackendSuite, test_atanh);
   //CU_ADD_TEST(onnxBackendSuite, test_atanh_example);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_1d_default);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_ceil);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_default);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_pads);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_pads_count_include_pad);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_precomputed_pads);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_precomputed_pads_count_include_pad);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_precomputed_same_upper);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_precomputed_strides);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_same_lower);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_same_upper);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_2d_strides);
   //CU_ADD_TEST(onnxBackendSuite, test_averagepool_3d_default);
   CU_ADD_TEST(onnxBackendSuite, test_basic_conv_with_padding);
   CU_ADD_TEST(onnxBackendSuite, test_basic_conv_without_padding);
   //CU_ADD_TEST(onnxBackendSuite, test_basic_convinteger);
   CU_ADD_TEST(onnxBackendSuite, test_batchnorm_epsilon);
   CU_ADD_TEST(onnxBackendSuite, test_batchnorm_example);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_left_uint16);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_left_uint32);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_left_uint64);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_left_uint8);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_right_uint16);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_right_uint32);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_right_uint64);
   //CU_ADD_TEST(onnxBackendSuite, test_bitshift_right_uint8);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_DOUBLE_to_FLOAT);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_DOUBLE_to_FLOAT16);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_FLOAT16_to_DOUBLE);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_FLOAT16_to_FLOAT);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_FLOAT_to_DOUBLE);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_FLOAT_to_FLOAT16);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_FLOAT_to_STRING);
   //CU_ADD_TEST(onnxBackendSuite, test_cast_STRING_to_FLOAT);
   //CU_ADD_TEST(onnxBackendSuite, test_ceil);
   //CU_ADD_TEST(onnxBackendSuite, test_ceil_example);
   //CU_ADD_TEST(onnxBackendSuite, test_clip);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_default_inbounds);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_default_max);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_default_min);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_example);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_inbounds);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_outbounds);
   //CU_ADD_TEST(onnxBackendSuite, test_clip_splitbounds);
   //CU_ADD_TEST(onnxBackendSuite, test_compress_0);
   //CU_ADD_TEST(onnxBackendSuite, test_compress_1);
   //CU_ADD_TEST(onnxBackendSuite, test_compress_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_compress_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_1d_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_1d_axis_negative_1);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_2d_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_2d_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_2d_axis_negative_1);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_2d_axis_negative_2);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_2);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_negative_1);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_negative_2);
   //CU_ADD_TEST(onnxBackendSuite, test_concat_3d_axis_negative_3);
   //CU_ADD_TEST(onnxBackendSuite, test_constant);
   //CU_ADD_TEST(onnxBackendSuite, test_constant_pad);
   //CU_ADD_TEST(onnxBackendSuite, test_constantofshape_float_ones);
   //CU_ADD_TEST(onnxBackendSuite, test_constantofshape_int_zeros);
   //CU_ADD_TEST(onnxBackendSuite, test_conv_with_strides_and_asymmetric_padding);
   CU_ADD_TEST(onnxBackendSuite, test_conv_with_strides_no_padding);
   CU_ADD_TEST(onnxBackendSuite, test_conv_with_strides_padding);
   //CU_ADD_TEST(onnxBackendSuite, test_convinteger_with_padding);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_1d);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_3d);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_dilations);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_kernel_shape);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_output_shape);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_pad);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_pads);
   //CU_ADD_TEST(onnxBackendSuite, test_convtranspose_with_kernel);
   //CU_ADD_TEST(onnxBackendSuite, test_cos);
   //CU_ADD_TEST(onnxBackendSuite, test_cos_example);
   //CU_ADD_TEST(onnxBackendSuite, test_cosh);
   //CU_ADD_TEST(onnxBackendSuite, test_cosh_example);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_1d);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_1d_exclusive);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_1d_reverse);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_1d_reverse_exclusive);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_2d_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_2d_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_cumsum_2d_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_depthtospace_crd_mode);
   //CU_ADD_TEST(onnxBackendSuite, test_depthtospace_crd_mode_example);
   //CU_ADD_TEST(onnxBackendSuite, test_depthtospace_dcr_mode);
   //CU_ADD_TEST(onnxBackendSuite, test_depthtospace_example);
   //CU_ADD_TEST(onnxBackendSuite, test_dequantizelinear);
   //CU_ADD_TEST(onnxBackendSuite, test_det_2d);
   //CU_ADD_TEST(onnxBackendSuite, test_det_nd);
   //CU_ADD_TEST(onnxBackendSuite, test_div);
   //CU_ADD_TEST(onnxBackendSuite, test_div_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_div_example);
   //CU_ADD_TEST(onnxBackendSuite, test_dropout_default);
   //CU_ADD_TEST(onnxBackendSuite, test_dropout_random);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear_max_adjusted);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear_max_adjusted_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear_min_adjusted);
   //CU_ADD_TEST(onnxBackendSuite, test_dynamicquantizelinear_min_adjusted_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_edge_pad);
   //CU_ADD_TEST(onnxBackendSuite, test_elu);
   //CU_ADD_TEST(onnxBackendSuite, test_elu_default);
   //CU_ADD_TEST(onnxBackendSuite, test_elu_example);
   //CU_ADD_TEST(onnxBackendSuite, test_equal);
   //CU_ADD_TEST(onnxBackendSuite, test_equal_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_erf);
   //CU_ADD_TEST(onnxBackendSuite, test_exp);
   //CU_ADD_TEST(onnxBackendSuite, test_exp_example);
   //CU_ADD_TEST(onnxBackendSuite, test_expand_dim_changed);
   //CU_ADD_TEST(onnxBackendSuite, test_expand_dim_unchanged);
   //CU_ADD_TEST(onnxBackendSuite, test_eyelike_populate_off_main_diagonal);
   //CU_ADD_TEST(onnxBackendSuite, test_eyelike_with_dtype);
   //CU_ADD_TEST(onnxBackendSuite, test_eyelike_without_dtype);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_axis0);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_axis1);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_axis2);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_axis3);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_negative_axis1);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_negative_axis2);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_negative_axis3);
   //CU_ADD_TEST(onnxBackendSuite, test_flatten_negative_axis4);
   //CU_ADD_TEST(onnxBackendSuite, test_floor);
   //CU_ADD_TEST(onnxBackendSuite, test_floor_example);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_0);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_1);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_elements_0);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_elements_1);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_elements_negative_indices);
   //CU_ADD_TEST(onnxBackendSuite, test_gather_negative_indices);
   //CU_ADD_TEST(onnxBackendSuite, test_gathernd_example_float32);
   //CU_ADD_TEST(onnxBackendSuite, test_gathernd_example_int32);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_all_attributes);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_alpha);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_beta);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_matrix_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_no_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_scalar_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_single_elem_vector_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_vector_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_default_zero_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_transposeA);
   //CU_ADD_TEST(onnxBackendSuite, test_gemm_transposeB);
   //CU_ADD_TEST(onnxBackendSuite, test_globalaveragepool);
   //CU_ADD_TEST(onnxBackendSuite, test_globalaveragepool_precomputed);
   //CU_ADD_TEST(onnxBackendSuite, test_globalmaxpool);
   //CU_ADD_TEST(onnxBackendSuite, test_globalmaxpool_precomputed);
   //CU_ADD_TEST(onnxBackendSuite, test_greater);
   //CU_ADD_TEST(onnxBackendSuite, test_greater_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_gru_defaults);
   //CU_ADD_TEST(onnxBackendSuite, test_gru_seq_length);
   //CU_ADD_TEST(onnxBackendSuite, test_gru_with_initial_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_axis_2);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_example);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_hardmax_one_hot);
   //CU_ADD_TEST(onnxBackendSuite, test_hardsigmoid);
   //CU_ADD_TEST(onnxBackendSuite, test_hardsigmoid_default);
   //CU_ADD_TEST(onnxBackendSuite, test_hardsigmoid_example);
   //CU_ADD_TEST(onnxBackendSuite, test_identity);
   //CU_ADD_TEST(onnxBackendSuite, test_instancenorm_epsilon);
   //CU_ADD_TEST(onnxBackendSuite, test_instancenorm_example);
   //CU_ADD_TEST(onnxBackendSuite, test_isinf);
   //CU_ADD_TEST(onnxBackendSuite, test_isinf_negative);
   //CU_ADD_TEST(onnxBackendSuite, test_isinf_positive);
   //CU_ADD_TEST(onnxBackendSuite, test_isnan);
   CU_ADD_TEST(onnxBackendSuite, test_leakyrelu);
   CU_ADD_TEST(onnxBackendSuite, test_leakyrelu_default);
   CU_ADD_TEST(onnxBackendSuite, test_leakyrelu_example);
   //CU_ADD_TEST(onnxBackendSuite, test_less);
   //CU_ADD_TEST(onnxBackendSuite, test_less_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_log);
   //CU_ADD_TEST(onnxBackendSuite, test_log_example);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_axis_2);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_example_1);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_large_number);
   //CU_ADD_TEST(onnxBackendSuite, test_logsoftmax_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_lrn);
   //CU_ADD_TEST(onnxBackendSuite, test_lrn_default);
   //CU_ADD_TEST(onnxBackendSuite, test_lstm_defaults);
   //CU_ADD_TEST(onnxBackendSuite, test_lstm_with_initial_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_lstm_with_peepholes);
   CU_ADD_TEST(onnxBackendSuite, test_matmul_2d);
   //CU_ADD_TEST(onnxBackendSuite, test_matmul_3d);
   //CU_ADD_TEST(onnxBackendSuite, test_matmul_4d);
   //CU_ADD_TEST(onnxBackendSuite, test_matmulinteger);
   //CU_ADD_TEST(onnxBackendSuite, test_max_example);
   //CU_ADD_TEST(onnxBackendSuite, test_max_one_input);
   //CU_ADD_TEST(onnxBackendSuite, test_max_two_inputs);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_1d_default);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_ceil);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_default);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_dilations);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_pads);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_precomputed_pads);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_precomputed_same_upper);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_precomputed_strides);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_same_lower);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_same_upper);
   CU_ADD_TEST(onnxBackendSuite, test_maxpool_2d_strides);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_3d_default);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_with_argmax_2d_precomputed_pads);
   //CU_ADD_TEST(onnxBackendSuite, test_maxpool_with_argmax_2d_precomputed_strides);
   //CU_ADD_TEST(onnxBackendSuite, test_maxunpool_export_with_output_shape);
   //CU_ADD_TEST(onnxBackendSuite, test_maxunpool_export_without_output_shape);
   //CU_ADD_TEST(onnxBackendSuite, test_mean_example);
   //CU_ADD_TEST(onnxBackendSuite, test_mean_one_input);
   //CU_ADD_TEST(onnxBackendSuite, test_mean_two_inputs);
   //CU_ADD_TEST(onnxBackendSuite, test_min_example);
   //CU_ADD_TEST(onnxBackendSuite, test_min_one_input);
   //CU_ADD_TEST(onnxBackendSuite, test_min_two_inputs);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_broadcast);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_int64_fmod);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_float16);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_float32);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_float64);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_int16);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_int32);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_int64);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_mixed_sign_int8);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_uint16);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_uint32);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_uint64);
   //CU_ADD_TEST(onnxBackendSuite, test_mod_uint8);
   //CU_ADD_TEST(onnxBackendSuite, test_mul);
   //CU_ADD_TEST(onnxBackendSuite, test_mul_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_mul_example);
   //CU_ADD_TEST(onnxBackendSuite, test_mvn);
   //CU_ADD_TEST(onnxBackendSuite, test_mvn_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_neg);
   //CU_ADD_TEST(onnxBackendSuite, test_neg_example);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_center_point_box_format);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_flipped_coordinates);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_identical_boxes);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_limit_output_size);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_single_box);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_suppress_by_IOU);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_suppress_by_IOU_and_scores);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_two_batches);
   //CU_ADD_TEST(onnxBackendSuite, test_nonmaxsuppression_two_classes);
   //CU_ADD_TEST(onnxBackendSuite, test_nonzero_example);
   //CU_ADD_TEST(onnxBackendSuite, test_not_2d);
   //CU_ADD_TEST(onnxBackendSuite, test_not_3d);
   //CU_ADD_TEST(onnxBackendSuite, test_not_4d);
   //CU_ADD_TEST(onnxBackendSuite, test_onehot_negative_indices);
   //CU_ADD_TEST(onnxBackendSuite, test_onehot_with_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_onehot_with_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_onehot_without_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_or2d);
   //CU_ADD_TEST(onnxBackendSuite, test_or3d);
   //CU_ADD_TEST(onnxBackendSuite, test_or4d);
   //CU_ADD_TEST(onnxBackendSuite, test_or_bcast3v1d);
   //CU_ADD_TEST(onnxBackendSuite, test_or_bcast3v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_or_bcast4v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_or_bcast4v3d);
   //CU_ADD_TEST(onnxBackendSuite, test_or_bcast4v4d);
   //CU_ADD_TEST(onnxBackendSuite, test_pow);
   //CU_ADD_TEST(onnxBackendSuite, test_pow_bcast_array);
   //CU_ADD_TEST(onnxBackendSuite, test_pow_bcast_scalar);
   //CU_ADD_TEST(onnxBackendSuite, test_pow_example);
   //CU_ADD_TEST(onnxBackendSuite, test_prelu_broadcast);
   //CU_ADD_TEST(onnxBackendSuite, test_prelu_example);
   //CU_ADD_TEST(onnxBackendSuite, test_qlinearconv);
   //CU_ADD_TEST(onnxBackendSuite, test_qlinearmatmul_2D);
   //CU_ADD_TEST(onnxBackendSuite, test_qlinearmatmul_3D);
   //CU_ADD_TEST(onnxBackendSuite, test_quantizelinear);
   //CU_ADD_TEST(onnxBackendSuite, test_range_float_type_positive_delta);
   //CU_ADD_TEST(onnxBackendSuite, test_range_float_type_positive_delta_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_range_int32_type_negative_delta);
   //CU_ADD_TEST(onnxBackendSuite, test_range_int32_type_negative_delta_expanded);
   //CU_ADD_TEST(onnxBackendSuite, test_reciprocal);
   //CU_ADD_TEST(onnxBackendSuite, test_reciprocal_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_keep_dims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_keep_dims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_negative_axes_keep_dims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l1_negative_axes_keep_dims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_keep_dims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_keep_dims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_negative_axes_keep_dims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_l2_negative_axes_keep_dims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_asc_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_default);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_desc_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_exp_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_log_sum_negative_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_default_axes_keepdim_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_max_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_mean_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_min_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_prod_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_default_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_default_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_do_not_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_do_not_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_negative_axes_keepdims_example);
   //CU_ADD_TEST(onnxBackendSuite, test_reduce_sum_square_negative_axes_keepdims_random);
   //CU_ADD_TEST(onnxBackendSuite, test_reflect_pad);
   CU_ADD_TEST(onnxBackendSuite, test_relu);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_extended_dims);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_negative_dim);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_negative_extended_dims);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_one_dim);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_reduced_dims);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_reordered_all_dims);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_reordered_last_dims);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_zero_and_negative_dim);
   //CU_ADD_TEST(onnxBackendSuite, test_reshape_zero_dim);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_cubic);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_cubic_A_n0p5_exclude_outside);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_cubic_align_corners);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_linear);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_linear_align_corners);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_scales_nearest);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_sizes_cubic);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_sizes_linear_pytorch_half_pixel);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_sizes_nearest);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_tf_crop_and_resize);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_cubic);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_cubic_A_n0p5_exclude_outside);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_cubic_align_corners);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_cubic_asymmetric);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_linear);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_linear_align_corners);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_scales_nearest);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_sizes_cubic);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_sizes_nearest);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_sizes_nearest_ceil_half_pixel);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_sizes_nearest_floor_align_corners);
   //CU_ADD_TEST(onnxBackendSuite, test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric);
   //CU_ADD_TEST(onnxBackendSuite, test_reversesequence_batch);
   //CU_ADD_TEST(onnxBackendSuite, test_reversesequence_time);
   //CU_ADD_TEST(onnxBackendSuite, test_rnn_seq_length);
   //CU_ADD_TEST(onnxBackendSuite, test_roialign);
   //CU_ADD_TEST(onnxBackendSuite, test_round);
   //CU_ADD_TEST(onnxBackendSuite, test_scan9_sum);
   //CU_ADD_TEST(onnxBackendSuite, test_scan_sum);
   //CU_ADD_TEST(onnxBackendSuite, test_scatter_elements_with_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_scatter_elements_with_negative_indices);
   //CU_ADD_TEST(onnxBackendSuite, test_scatter_elements_without_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_scatter_with_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_scatter_without_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_scatternd);
   //CU_ADD_TEST(onnxBackendSuite, test_selu);
   //CU_ADD_TEST(onnxBackendSuite, test_selu_default);
   //CU_ADD_TEST(onnxBackendSuite, test_selu_example);
   //CU_ADD_TEST(onnxBackendSuite, test_shape);
   //CU_ADD_TEST(onnxBackendSuite, test_shape_example);
   //CU_ADD_TEST(onnxBackendSuite, test_shrink_hard);
   //CU_ADD_TEST(onnxBackendSuite, test_shrink_soft);
   //CU_ADD_TEST(onnxBackendSuite, test_sigmoid);
   //CU_ADD_TEST(onnxBackendSuite, test_sigmoid_example);
   //CU_ADD_TEST(onnxBackendSuite, test_sign);
   //CU_ADD_TEST(onnxBackendSuite, test_simple_rnn_defaults);
   //CU_ADD_TEST(onnxBackendSuite, test_simple_rnn_with_initial_bias);
   //CU_ADD_TEST(onnxBackendSuite, test_sin);
   //CU_ADD_TEST(onnxBackendSuite, test_sin_example);
   //CU_ADD_TEST(onnxBackendSuite, test_sinh);
   //CU_ADD_TEST(onnxBackendSuite, test_sinh_example);
   //CU_ADD_TEST(onnxBackendSuite, test_size);
   //CU_ADD_TEST(onnxBackendSuite, test_size_example);
   //CU_ADD_TEST(onnxBackendSuite, test_slice);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_default_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_default_steps);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_end_out_of_bounds);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_neg);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_neg_steps);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_negative_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_slice_start_out_of_bounds);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_axis_2);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_example);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_large_number);
   //CU_ADD_TEST(onnxBackendSuite, test_softmax_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_softplus);
   //CU_ADD_TEST(onnxBackendSuite, test_softplus_example);
   //CU_ADD_TEST(onnxBackendSuite, test_softsign);
   //CU_ADD_TEST(onnxBackendSuite, test_softsign_example);
   //CU_ADD_TEST(onnxBackendSuite, test_split_equal_parts_1d);
   //CU_ADD_TEST(onnxBackendSuite, test_split_equal_parts_2d);
   //CU_ADD_TEST(onnxBackendSuite, test_split_equal_parts_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_split_variable_parts_1d);
   //CU_ADD_TEST(onnxBackendSuite, test_split_variable_parts_2d);
   //CU_ADD_TEST(onnxBackendSuite, test_split_variable_parts_default_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_sqrt);
   //CU_ADD_TEST(onnxBackendSuite, test_sqrt_example);
   //CU_ADD_TEST(onnxBackendSuite, test_squeeze);
   //CU_ADD_TEST(onnxBackendSuite, test_squeeze_negative_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_export_monday_casesensintive_lower);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_export_monday_casesensintive_nochangecase);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_export_monday_casesensintive_upper);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_export_monday_empty_output);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_export_monday_insensintive_upper_twodim);
   //CU_ADD_TEST(onnxBackendSuite, test_strnormalizer_nostopwords_nochangecase);
   //CU_ADD_TEST(onnxBackendSuite, test_sub);
   //CU_ADD_TEST(onnxBackendSuite, test_sub_bcast);
   //CU_ADD_TEST(onnxBackendSuite, test_sub_example);
   //CU_ADD_TEST(onnxBackendSuite, test_sum_example);
   //CU_ADD_TEST(onnxBackendSuite, test_sum_one_input);
   //CU_ADD_TEST(onnxBackendSuite, test_sum_two_inputs);
   //CU_ADD_TEST(onnxBackendSuite, test_tan);
   //CU_ADD_TEST(onnxBackendSuite, test_tan_example);
   //CU_ADD_TEST(onnxBackendSuite, test_tanh);
   //CU_ADD_TEST(onnxBackendSuite, test_tanh_example);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_batch_onlybigrams_skip0);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_batch_onlybigrams_skip5);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_batch_uniandbigrams_skip5);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_only_bigrams_skip0);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_onlybigrams_levelempty);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_onlybigrams_skip5);
   //CU_ADD_TEST(onnxBackendSuite, test_tfidfvectorizer_tf_uniandbigrams_skip5);
   //CU_ADD_TEST(onnxBackendSuite, test_thresholdedrelu);
   //CU_ADD_TEST(onnxBackendSuite, test_thresholdedrelu_default);
   //CU_ADD_TEST(onnxBackendSuite, test_thresholdedrelu_example);
   //CU_ADD_TEST(onnxBackendSuite, test_tile);
   //CU_ADD_TEST(onnxBackendSuite, test_tile_precomputed);
   //CU_ADD_TEST(onnxBackendSuite, test_top_k);
   //CU_ADD_TEST(onnxBackendSuite, test_top_k_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_top_k_smallest);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_0);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_1);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_2);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_3);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_4);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_all_permutations_5);
   //CU_ADD_TEST(onnxBackendSuite, test_transpose_default);
   //CU_ADD_TEST(onnxBackendSuite, test_unique_not_sorted_without_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_unique_sorted_with_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_unique_sorted_with_axis_3d);
   //CU_ADD_TEST(onnxBackendSuite, test_unique_sorted_with_negative_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_unique_sorted_without_axis);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_axis_0);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_axis_1);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_axis_2);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_axis_3);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_negative_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_three_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_two_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_unsqueeze_unsorted_axes);
   //CU_ADD_TEST(onnxBackendSuite, test_upsample_nearest);
   //CU_ADD_TEST(onnxBackendSuite, test_where_example);
   //CU_ADD_TEST(onnxBackendSuite, test_where_long_example);
   //CU_ADD_TEST(onnxBackendSuite, test_xor2d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor3d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor4d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor_bcast3v1d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor_bcast3v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor_bcast4v2d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor_bcast4v3d);
   //CU_ADD_TEST(onnxBackendSuite, test_xor_bcast4v4d);


  /* Models tests */
  /* This suite tests a whole model end to end. All data is taken
   * form ONNX repository, where the model plus a set of inputs/outputs
   * is provided */
  CU_add_test(modelsTestSuite, "test_model_mnist", test_model_mnist);
  //CU_add_test(modelsTestSuite, "test_model_mnist_per_node", test_model_mnist_per_node);
  CU_add_test(modelsTestSuite, "test_model_tinyyolov2", test_model_tinyyolov2);
  //CU_add_test(modelsTestSuite, "test_model_super_resolution", test_model_super_resolution);

  CU_basic_set_mode(CU_BRM_VERBOSE);

  // If not inputs are provided, run everything
  if (argc == 1){
    CU_basic_run_tests();
  }

  else if (argc == 2){
    printf("running specific ts\n");
    CU_pSuite suite2run = CU_get_suite(argv[1]);
    CU_basic_run_suite(suite2run);
  }else if (argc == 3){
    printf("running specific tc from a ts\n");
    CU_pSuite suite2run = CU_get_suite(argv[1]);
    CU_pTest test2run = CU_get_test(suite2run, argv[2]);
    CU_basic_run_test(suite2run, test2run);
  }

  printf("CU_get_number_of_tests_run is %d\n", CU_get_number_of_tests_run());
  printf("CU_get_number_of_tests_failed is %d\n", CU_get_number_of_tests_failed());
  // TODO Temporal hackish solution to force CI to fail
  if (CU_get_number_of_tests_failed() != 0){
    exit (1);
  }

  CU_cleanup_registry();
  return CU_get_error();
}

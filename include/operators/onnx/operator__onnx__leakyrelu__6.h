//this file was generated by ../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__ONNX__LEAKYRELU__6_H
# define OPERATOR_OPERATOR__ONNX__LEAKYRELU__6_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * onnx operator 'LeakyRelu' version 6
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
 * output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
 * `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T X:
 *   Input tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T Y:
 *   Output tensor
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute FLOAT alpha :
 *   Coefficient of leakage.
 *
 * @since version 6
 *
 * @see home/drechsler/git/cONNXr/third_party/onnx/onnx/onnx/defs/math/defs.cc:328
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu
 */
extern __attribute__((weak))
operator_status operator__onnx__leakyrelu__6(
    node_context *ctx
);

operator_executer resolve_operator__onnx__leakyrelu__6(
    node_context *ctx
);

extern __attribute__((weak)) operator_info info_operator__onnx__leakyrelu__6;

extern __attribute__((weak))
operator_status operator__onnx__leakyrelu__6__T_tensor_double(
    node_context *ctx
);
extern __attribute__((weak))
operator_status operator__onnx__leakyrelu__6__T_tensor_float(
    node_context *ctx
);
extern __attribute__((weak))
operator_status operator__onnx__leakyrelu__6__T_tensor_float16(
    node_context *ctx
);
# endif
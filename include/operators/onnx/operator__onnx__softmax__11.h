//this file was generated by ../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__ONNX__SOFTMAX__11_H
# define OPERATOR_OPERATOR__ONNX__SOFTMAX__11_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * onnx operator 'Softmax' version 11
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * The operator computes the softmax (normalized exponential) values for each layer in the batch
 *  of the given input.
 * 
 * The input does not need to explicitly be a 2D vector; rather, it will be
 * coerced into one. For an arbitrary n-dimensional tensor
 * input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
 * the axis provided, then input will be coerced into a 2-dimensional tensor with
 * dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
 * case where axis=1, this means the input tensor will be coerced into a 2D tensor
 * of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
 * In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
 * Each of these dimensions must be matched correctly, or else the operator
 * will throw errors. The output tensor has the same shape
 * and contains the softmax values of the corresponding input.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Input T input:
 *   The input tensor that's coerced into a 2D matrix of size (NxD) as
 *   described above.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Output T output:
 *   The output values with the same shape as input tensor (the original size
 *   without coercion).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute INT axis :
 *   Describes the axis of the inputs when coerced to 2D; defaults to one
 *   because the 0th axis most likely describes the batch_size. Negative value
 *   means counting dimensions from the back. Accepted range is [-r, r-1] where
 *   r = rank(input).
 *
 * @since version 11
 *
 * @see io/onnx/onnx/defs/math/defs.cc:783
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
 */
__attribute__((weak))
operator_status operator__onnx__softmax__11(
    node_context *ctx
);

operator_executer resolve_operator__onnx__softmax__11(
    node_context *ctx
);

extern __attribute__((weak)) operator_info info_operator__onnx__softmax__11;

__attribute__((weak))
operator_status operator__onnx__softmax__11__T_tensor_double(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__softmax__11__T_tensor_float(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__softmax__11__T_tensor_float16(
    node_context *ctx
);
# endif
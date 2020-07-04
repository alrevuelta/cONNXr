//this file was generated by ../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__ONNX__ARGMAX__12_H
# define OPERATOR_OPERATOR__ONNX__ARGMAX__12_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * onnx operator 'ArgMax' version 12
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Computes the indices of the max elements of the input tensor's element along the 
 * provided axis. The resulting tensor has the same rank as the input if keepdims equal 1. 
 * If keepdims equal 0, then the resulting tensor have the reduced dimension pruned. 
 * If select_last_index is True (default False), the index of the last occurrence of the max 
 * is selected if the max appears more than once in the input. Otherwise the index of the 
 * first occurrence is selected.
 * The type of the output tensor is integer.
 * 
 * Constraint T:
 *   Constrain input and output types to all numeric tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_uint16,
 *                  tensor_uint32, tensor_uint64, tensor_uint8
 * Input T data:
 *   An input tensor.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int16,
 *                  tensor_int32, tensor_int64, tensor_int8, tensor_uint16,
 *                  tensor_uint32, tensor_uint64, tensor_uint8
 * Output tensor(int64) reduced:
 *   Reduced output tensor with integer data type.
 *   Allowed Types: tensor_int64
 * Attribute INT axis :
 *   The axis in which to compute the arg indices. Accepted range is [-r, r-1]
 *   where r = rank(data).
 * 
 * Attribute INT keepdims :
 *   Keep the reduced dimension or not, default 1 mean keep reduced dimension.
 * 
 * Attribute INT select_last_index :
 *   Whether to select the last index or the first index if the {name} appears
 *   in multiple indices, default is False (first index).
 *
 * @since version 12
 *
 * @see io/onnx/onnx/defs/reduction/defs.cc:238
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
 */
__attribute__((weak))
operator_status operator__onnx__argmax__12(
    node_context *ctx
);

operator_executer resolve_operator__onnx__argmax__12(
    node_context *ctx
);

extern __attribute__((weak)) operator_info info_operator__onnx__argmax__12;

__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_double(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_float(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_float16(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_int16(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_int32(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_int64(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_int8(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_uint16(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_uint32(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_uint64(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__argmax__12__T_tensor_uint8(
    node_context *ctx
);
# endif
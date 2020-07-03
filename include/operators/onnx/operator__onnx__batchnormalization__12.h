//this file was generated by ../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__ONNX__BATCHNORMALIZATION__12_H
# define OPERATOR_OPERATOR__ONNX__BATCHNORMALIZATION__12_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * onnx operator 'BatchNormalization' version 12
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * Carries out batch normalization as described in the paper https://arxiv.org/abs/1502.03167.
 * There is three required inputs 'X', 'mean' and 'var', in addition to one optional input 'training_mode'.
 * Note that 'mean' and 'var' are expected to be the estimated statistics in inference mode (training_mode=False, default),
 * and the running statistics in training mode (traning_mode=True).
 * There is one required output 'Y' and four optional outputs : 'output_mean', 'output_var', 'saved_mean', 'saved_var' used for training.
 * 
 * The output and statistics are updated as follows when training_mode=True:
 * ```
 * saved_mean = ReducedMean(X, axis=all_except_channel_index)
 * saved_var =  ReducedVar(X, axis=all_except_channel_index)
 * 
 * output_mean = mean * momentum + saved_mean * (1 - momentum)
 * output_var = var * momentum + saved_var * (1 - momentum)
 * 
 * Y = (X - saved_mean) / sqrt(var + saved_epsilon) * scale + B
 * ```
 * 
 * When training_mode=False:
 * ```
 * saved_mean = ReducedMean(X, axis=all_except_channel_index)
 * saved_var =  ReducedVar(X, axis=all_except_channel_index)
 * 
 * output_mean = mean
 * output_var = var
 * 
 * Y = (X - mean) / sqrt(var + epsilon) * scale + B
 * ```
 * 
 * For previous (depreciated) non-spatial cases, implementors are suggested
 * to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization operator.
 * This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
 * 
 * Constraint T:
 *   Constrain input and output types to float tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Constraint T1:
 *   Constrain input 'training_mode' types to boolean tensors.
 *   Allowed Types: tensor_bool
 * Input T X:
 *   Input data tensor from the previous operator; dimensions are in the form
 *   of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of
 *   channels. Statistics are computed for every channel of C over N and D1 to
 *   Dn dimensions. For image data, input dimensions become (N x C x H x W).
 *   The op also accepts single dimension input of size N in which case C is
 *   assumed to be 1
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T scale:
 *   Scale tensor of shape (C).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T B:
 *   Bias tensor of shape (C).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T mean:
 *   running (training) or estimated (testing) mean tensor of shape (C).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T var:
 *   running (training) or estimated (testing) variance tensor of shape (C).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Input T1 training_mode:
 *   If set to true, run spatial batch normalization in training mode, default
 *   is false.
 *   Allowed Types: tensor_bool
 * Output T Y:
 *   The output tensor of the same shape as X
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output T output_mean:
 *   The running mean when training_mode=True, or the estimated mean when
 *   training_mode=False (Tensor of shape (C)).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output T output_var:
 *   The running variance when training_mode=True, or the estimated variance
 *   when training_mode=False (Tensor of shape (C)).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output T saved_mean:
 *   Saved mean used during training to speed up gradient computation (Tensor
 *   of shape (C)).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * 
 * Output T saved_var:
 *   Saved variance used during training to speed up gradient computation
 *   (Tensor of shape (C)).
 *   Allowed Types: tensor_double, tensor_float, tensor_float16
 * Attribute FLOAT epsilon :
 *   The epsilon value to use to avoid division by zero.
 * 
 * Attribute FLOAT momentum :
 *   Factor used in computing the running mean and variance.e.g., output_mean
 *   = mean * momentum + saved_mean * (1 - momentum).
 *
 * @since version 12
 *
 * @see home/drechsler/git/cONNXr/third_party/onnx/onnx/onnx/defs/nn/defs.cc:1527
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
 */
extern __attribute__((weak))
operator_status operator__onnx__batchnormalization__12(
    node_context *ctx
);

operator_executer resolve_operator__onnx__batchnormalization__12(
    node_context *ctx
);

extern __attribute__((weak)) operator_info info_operator__onnx__batchnormalization__12;

extern __attribute__((weak))
operator_status operator__onnx__batchnormalization__12__T_tensor_double__T1_tensor_bool(
    node_context *ctx
);
extern __attribute__((weak))
operator_status operator__onnx__batchnormalization__12__T_tensor_float__T1_tensor_bool(
    node_context *ctx
);
extern __attribute__((weak))
operator_status operator__onnx__batchnormalization__12__T_tensor_float16__T1_tensor_bool(
    node_context *ctx
);
# endif
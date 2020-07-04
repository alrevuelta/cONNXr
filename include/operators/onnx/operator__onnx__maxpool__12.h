//this file was generated by ../../../../scripts/onnx_generator/OperatorHeader.py
# ifndef OPERATOR_OPERATOR__ONNX__MAXPOOL__12_H
# define OPERATOR_OPERATOR__ONNX__MAXPOOL__12_H

# include "operators/operator.h"
# include "operators/operator_stub.h"
# include "operators/operator_info.h"

/**
 * onnx operator 'MaxPool' version 12
 *
 * @param[in]  ctx  Operator context
 * @return          Status code
 *
 * MaxPool consumes an input tensor X and applies max pooling across
 *  the tensor according to kernel sizes, stride sizes, and pad lengths.
 *  max pooling consisting of computing the max on all values of a
 *  subset of the input tensor according to the kernel size and downsampling the
 *  data into the output tensor Y for further processing. The output spatial shape will be following:
 *  ```
 *  output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 *  ```
 *  or
 *  ```
 *  output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 *  ```
 *  if ceil_mode is enabled
 * 
 *  ```
 *  * pad_shape[i] is sum of pads along axis i
 *  ```
 * 
 *  `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 *  ```
 *  VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 *  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 *  ```
 *  And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 *  ```
 *  pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 *  ```
 *  The output of each pooling window is maximum number of elements exclude pad.
 * 
 * Constraint T:
 *   Constrain input and output types to float and 8 bit tensors.
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int8,
 *                  tensor_uint8
 * 
 * Constraint I:
 *   Constrain index tensor to int64
 *   Allowed Types: tensor_int64
 * Input T X:
 *   Input data tensor from the previous operator; dimensions for image case
 *   are (N x C x H x W), where N is the batch size, C is the number of
 *   channels, and H and W are the height and the width of the data. For non
 *   image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
 *   where N is the batch size. Optionally, if dimension denotation is in
 *   effect, the operation expects the input data tensor to arrive with the
 *   dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
 *   DATA_FEATURE ...].
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int8,
 *                  tensor_uint8
 * Output T Y:
 *   Output data tensor from average or max pooling across the input tensor.
 *   Dimensions will vary based on various kernel, stride, and pad sizes. Floor
 *   value of the dimension is used
 *   Allowed Types: tensor_double, tensor_float, tensor_float16, tensor_int8,
 *                  tensor_uint8
 * 
 * Output I Indices:
 *   Indices tensor from max pooling across the input tensor. The dimensions
 *   of indices are the same as output tensor. The values in indices of are the
 *   indices of the selected values during pooling. The indices are computed as
 *   flatten 1-D tensor, and the indices do not consider padding. So the values
 *   in indices are in [0, N x C x D1 x ... x Dn).
 *   Allowed Types: tensor_int64
 * Attribute STRING auto_pad :
 *   auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where
 *   default value is NOTSET, which means explicit padding is used. SAME_UPPER
 *   or SAME_LOWER mean pad the input so that the output spatial size match the
 *   input.In case of odd number add the extra padding at the end for
 *   SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding.
 * 
 * Attribute INT ceil_mode :
 *   Whether to use ceil or floor (default) to compute the output shape.
 * 
 * Attribute INTS dilations :
 *   Dilation value along each spatial axis of filter. If not present, the
 *   dilation defaults to 1 along each spatial axis.
 * 
 * Attribute INTS kernel_shape (optional):
 *   The size of the kernel along each axis.
 * 
 * Attribute INTS pads :
 *   Padding for the beginning and ending along each spatial axis, it can take
 *   any value greater than or equal to 0. The value represent the number of
 *   pixels added to the beginning and end part of the corresponding axis.
 *   `pads` format should be as follow [x1_begin, x2_begin...x1_end,
 *   x2_end,...], where xi_begin the number of pixels added at the beginning of
 *   axis `i` and xi_end, the number of pixels added at the end of axis `i`.
 *   This attribute cannot be used simultaneously with auto_pad attribute. If
 *   not present, the padding defaults to 0 along start and end of each spatial
 *   axis.
 * 
 * Attribute INT storage_order :
 *   The storage order of the tensor. 0 is row major, and 1 is column major.
 * 
 * Attribute INTS strides :
 *   Stride along each spatial axis. If not present, the stride defaults to 1
 *   along each spatial axis.
 *
 * @since version 12
 *
 * @see io/onnx/onnx/defs/nn/defs.cc:363
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
 */
__attribute__((weak))
operator_status operator__onnx__maxpool__12(
    node_context *ctx
);

operator_executer resolve_operator__onnx__maxpool__12(
    node_context *ctx
);

extern __attribute__((weak)) operator_info info_operator__onnx__maxpool__12;

__attribute__((weak))
operator_status operator__onnx__maxpool__12__T_tensor_double(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__maxpool__12__T_tensor_float(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__maxpool__12__T_tensor_float16(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__maxpool__12__T_tensor_int8(
    node_context *ctx
);
__attribute__((weak))
operator_status operator__onnx__maxpool__12__T_tensor_uint8(
    node_context *ctx
);
# endif
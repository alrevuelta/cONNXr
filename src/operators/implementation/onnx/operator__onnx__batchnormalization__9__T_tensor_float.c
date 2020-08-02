#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"

operator_status operator__onnx__batchnormalization__9__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);
  TRACE_NODE(2, true, ctx->onnx_node);

  Onnx__TensorProto *X = searchInputByName(ctx, 0);
  Onnx__TensorProto *scale = searchInputByName(ctx, 1);
  Onnx__TensorProto *B = searchInputByName(ctx, 2);
  Onnx__TensorProto *mean = searchInputByName(ctx, 3);
  Onnx__TensorProto *var = searchInputByName(ctx, 4);

  TRACE_TENSOR(2, true, X);
  TRACE_TENSOR(2, true, scale);
  TRACE_TENSOR(2, true, B);
  TRACE_TENSOR(2, true, mean);
  TRACE_TENSOR(2, true, var);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  /* TODO: Not implemented
  Onnx__TensorProto *mean = searchOutputByName(ctx, 1);
  Onnx__TensorProto *var = searchOutputByName(ctx, 2);
  Onnx__TensorProto *saved_mean = searchOutputByName(ctx, 3);
  Onnx__TensorProto *saved_var = searchOutputByName(ctx, 4);*/

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  //Onnx__AttributeProto *momentumAttr = searchAttributeNyName(n_attribute, attribute, "momentum");
  /* Epsilon is hardcoded to float */
  float eps = 0.00001; /* Default value */

  if (ctx->onnx_node->n_attribute == 1){
    eps =ctx->onnx_node->attribute[0]->f;
  }

  // Allocte memory
  Y->dims = malloc(X->n_dims * sizeof(int64_t));
  Y->n_dims = X->n_dims;
  for(int i = 0; i < Y->n_dims; i++){
    Y->dims[i] = X->dims[i];
  }
  Y->has_raw_data = 0;
  Y->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  Y->n_float_data = X->n_float_data;
  Y->float_data   = malloc(Y->n_float_data * sizeof(float));

  for (int i = 0; i < Y->n_float_data; i++) {
    TRACE_BOUND(3, true, i, 0, (int)Y->n_float_data, "%d");
    int index = (i/(X->dims[2] * X->dims[3])) % X->dims[1];
    TRACE_VAR(3, true, X->float_data[i], "%f");
    TRACE_VAR(3, true, index, "%d");
    TRACE_VAR(3, true, mean->float_data[index], "%f");
    TRACE_VAR(3, true, var->float_data[index], "%f");
    TRACE_VAR(3, true, scale->float_data[index], "%f");
    TRACE_VAR(3, true, B->float_data[index], "%f");
    Y->float_data[i] = (X->float_data[i] - mean->float_data[index]) / sqrtf(var->float_data[index] + eps);
    Y->float_data[i] = scale->float_data[index] * Y->float_data[i] + B->float_data[index];
    TRACE_VAR(3, true, Y->float_data[i], "%f");
  }

  TRACE_TENSOR(2, true, Y);
  TRACE_EXIT(1);

  return 0;
}

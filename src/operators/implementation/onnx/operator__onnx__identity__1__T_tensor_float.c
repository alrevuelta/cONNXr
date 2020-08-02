#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"

operator_status operator__onnx__identity__1__T_tensor_float(
 node_context *ctx
)
{
  TRACE_ENTRY(1);

  Onnx__TensorProto *X = searchInputByName(ctx, 0);

  TRACE_TENSOR(2, true, X);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  if (0){
   /* TODO: Check some conditions. For example if a specific
    * functionality is not supported */
   return -1;
  }


  Y->dims = malloc(X->n_dims * sizeof(int64_t));
  for (int i = 0; i < X->n_dims; i++)
  {
   Y->dims[i] = X->dims[i];
  }

  // Populate some parameters
  Y->n_dims       = X->n_dims;
  Y->has_raw_data = 0;
  Y->data_type    = X->data_type;

  Y->n_float_data = X->n_float_data;
  Y->float_data = malloc(Y->n_float_data * sizeof(float));
  for (int i = 0; i < Y->n_float_data; i++)
  {
   Y->float_data[i] = X->float_data[i];
  }

  TRACE_TENSOR(2, true, Y);
  TRACE_EXIT(1);
  return 0;
}

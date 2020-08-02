#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"

operator_status operator__onnx__matmul__9__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);

  TRACE_TENSOR(2, true, A);
  TRACE_TENSOR(2, true, B);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return -1;
  }

  // TODO Hardcoded for 2 dimensions
  // TODO Might be useful to define a macro like
  // #define I(a,b,c,d) I[(a)+(b)*oH+(c)*oH*oW+(d)*oH*oW*C]
  // dont know how to handle the different dimensions though

  // Allocte memory
  Y->dims = malloc(2 * sizeof(int64_t));

  // Populate some parameters
  Y->n_dims       = 2;
  Y->dims[0]      = A->dims[0];
  Y->dims[1]      = B->dims[1];
  Y->has_raw_data = 0;
  Y->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  Y->n_float_data = A->dims[0] * B->dims[1];
  Y->float_data   = malloc(A->dims[0] * B->dims[1] * sizeof(float));

  for (int i = 0; i < A->dims[0]; i++) {
    for (int j = 0; j < B->dims[1]; j++) {
      float sum = 0;
      for (int p = 0; p < A->dims[1]; p++) {
        sum += (A->float_data[i*A->dims[1]+p] * B->float_data[p*B->dims[1]+j]);
        // Saturate the value?
      }
      Y->float_data[i*B->dims[1]+j] = sum;
    }
  }

  /* TODO Create new function for INT32
  Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT32;
  Y->n_int32_data = A->dims[0] * B->dims[1];
  Y->int32_data = malloc(A->dims[0] * B->dims[1] * sizeof(int32_t));
  for (int i = 0; i < A->dims[0]; i++) {
    for (int j = 0; j < B->dims[1]; j++) {
      int32_t sum = 0;
      for (int p = 0; p < A->dims[1]; p++) {
        sum += (A->int32_data[i*A->dims[1]+p] * B->int32_data[p*B->dims[1]+j]);
        // Saturate the value?
      }
      Y->int32_data[i*B->dims[1]+j] = sum;
    }
  }*/

  TRACE_TENSOR(2, true, Y);
  TRACE_EXIT(1);

  return 0;
}

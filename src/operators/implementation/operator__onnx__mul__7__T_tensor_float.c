#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"
#include "utils.h"

 operator_status operator__onnx__mul__7__T_tensor_float(
     node_context *ctx
 )
{
  TRACE_LEVEL0("Calling operator_mul\n");

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);

  Onnx__TensorProto *C = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  /* TODO: Hardcoded for tiny YOLO */

  /* Move this block to a common function */
  C->dims   = malloc(A->n_dims * sizeof(int64_t));
  C->n_dims = A->n_dims;

  for (int i = 0; i < A->n_dims; i++)
  {
    C->dims[i] = A->dims[i];
  }
  C->has_raw_data = 0;
  C->data_type = A->data_type;

  C->n_float_data = A->n_float_data;
  C->float_data = malloc(C->n_float_data * sizeof(float));

  for (int i = 0; i < A->n_float_data; i++){
    C->float_data[i] = A->float_data[i] * B->float_data[0];
  }
  return 0;
}

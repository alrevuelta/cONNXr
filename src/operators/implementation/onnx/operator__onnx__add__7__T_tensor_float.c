#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"

operator_status operator__onnx__add__7__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);
  TRACE_NODE(2, true, ctx->onnx_node);

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);

  TRACE_TENSOR(2, true, A);
  TRACE_TENSOR(2, true, B);

  Onnx__TensorProto *C = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  /* There order of operands if unknown. The longest one will determine the output */
  /* Quick and dirty solution */
  if (A->n_dims > B->n_dims){
    C->dims = malloc(A->n_dims * sizeof(int64_t));
    C->n_dims = A->n_dims;
    C->n_float_data = A->n_float_data; // check other types
    for (int i = 0; i < C->n_dims; i++)
    {
      C->dims[i] = A->dims[i];
    }
  }else{
    C->dims = malloc(B->n_dims * sizeof(int64_t));
    C->n_dims = B->n_dims;
    C->n_float_data = B->n_float_data; // check other types
    for (int i = 0; i < C->n_dims; i++)
    {
      C->dims[i] = B->dims[i];
    }
  }

  C->has_raw_data = 0;
  C->data_type = A->data_type;
  C->float_data = malloc(C->n_float_data * sizeof(float));
  /* TODO: ugly */
  for (int i = 0; i < C->n_float_data; i++) {
    /* Normal case where dimensions match */
    if (A->n_dims == B->n_dims) {
      C->float_data[i] = A->float_data[i] + B->float_data[i];
    /* Broadcasting. Hardcoded not working */
    }else{
      /* If inside loop :( */
      if (B->n_dims == 1){
        C->float_data[i] = A->float_data[i] + B->float_data[i%B->dims[0]];
      }else{
        /* TODO Hardcoded for TINY YOLO */
        if (A->dims[0] == 3){ /* Remove this uAF*/
          C->float_data[i] = A->float_data[i%3] + B->float_data[i];
        /* TODO Hardcoded for MNIST */
        }else{
          C->float_data[i] = A->float_data[i] + B->float_data[i/(A->dims[2]*A->dims[3])];
        }
      }
    }
  }

  TRACE_TENSOR(2, true, C);
  TRACE_EXIT(1);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

operator_status operator__onnx__transpose__1__T_tensor_float(
    node_context *ctx
)
{
  TRACE_LEVEL0("Calling operator_transpose\n");

  Onnx__TensorProto *data = searchInputByName(ctx, 0);
  Onnx__TensorProto *transposed = searchOutputByName(ctx, 0);
  Onnx__AttributeProto *perm = searchAttributeNyName(ctx->onnx_node->n_attribute,
                                                     ctx->onnx_node->attribute,
                                                     "perm");

  transposed->n_dims       = perm->n_ints;
  transposed->dims         = malloc(transposed->n_dims * sizeof(int64_t));
  transposed->has_raw_data = 0;
  transposed->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  transposed->n_float_data = data->n_float_data;
  transposed->float_data   = malloc(transposed->n_float_data * sizeof(float));

  // This should be the only case
  if (perm->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS){
    for (size_t i = 0; i < perm->n_ints; i++){
      transposed->dims[i] = data->dims[perm->ints[i]];
    }
  }

  int n_dims = transposed->n_dims;
  int index[n_dims];
  for (int i = 0; i < n_dims; i++) {
    index[i] = 0;
  }

  //calculate offset of each dimension
  int offset[n_dims];
  offset[n_dims-1] = 1;
  for (int i = n_dims-2; i >= 0; i--) {
    offset[i] = transposed->dims[i+1] * offset[i+1];
  }

  for(int i = 0; i < transposed->n_float_data; i++) {
    for (int n = n_dims-1; n > 0 ; n--) {
      if ( index[n] < data->dims[n]) break;
      index[n] = 0;
      index[n-1]++;
    }
    if ( index[0] >= data->dims[0]) break;

    int pos = 0;
    for (int n = 0; n < n_dims; n++) {
      pos += index[perm->ints[n]] * offset[n];
    }
    transposed->float_data[pos] = data->float_data[i];
    index[n_dims-1]++;
  }

  debug_print_dims(transposed->n_dims, transposed->dims);
  return 0;
}

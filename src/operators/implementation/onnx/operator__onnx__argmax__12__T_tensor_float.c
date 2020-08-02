#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"

operator_status operator__onnx__argmax__12__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);
  TRACE_NODE(2, true, ctx->onnx_node);

  Onnx__TensorProto *axis = searchInputByName(ctx, 0);
  //Onnx__TensorProto *keepdims = searchInputByName(ctx, 1);
  //Onnx__TensorProto *select_last_index = searchInputByName(ctx, 2);

  TRACE_TENSOR(2, true, axis);

  Onnx__TensorProto *data = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // TODO axis and keepdims is not implemented
    // TODO Only a simple case with 2x2 matrix is implemented
    // TODO in all operators, init unused fields.
    // i.e. if the tensor is int64, init n_float_data to 0
    return 1;
  }

  // Allocte memory
  data->dims = malloc(axis->dims[1] * sizeof(int64_t));

  // Populate some parameters
  data->n_dims       = 1;
  data->dims[0]      = axis->dims[1];
  data->has_raw_data = 0;

  // INT64 is hardcoded by design
  data->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
  data->n_int64_data = axis->dims[1];
  data->int64_data = malloc(data->n_int64_data * sizeof(int64_t));
  for (int i = 0; i < axis->dims[1]; i++)
  {
    // init maxval to the first elemen
    float maxval = axis->float_data[i];
    int64_t maxind = 0;
    // todo start from j = 1?
    for (int j = 0; j < axis->dims[0]; j++)
    {
      if (axis->float_data[i + axis->dims[1] * j] > maxval)
      {
        maxind = j;
        maxval = axis->float_data[i + axis->dims[1] * j];
      }
    }
    data->int64_data[i] = maxind;
  }

  TRACE_EXIT(1);
  TRACE_TENSOR(2, true, data);

  return 0;
}

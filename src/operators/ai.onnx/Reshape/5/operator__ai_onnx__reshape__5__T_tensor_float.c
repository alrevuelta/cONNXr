#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "tracing.h"
#include "utils.h"
#include "operators/ai.onnx/Reshape/5/operator__ai_onnx__reshape__5.h"


operator_status
operator__ai_onnx__reshape__5__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);

  Onnx__TensorProto *data = searchInputByName(ctx, 0);
  Onnx__TensorProto *shape = searchInputByName(ctx, 1);
  Onnx__TensorProto *reshaped = searchOutputByName(ctx, 0);

  TRACE_TENSOR(2,true,data);
  TRACE_TENSOR(2,true,shape);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  // Not sure about this implementation. It just swaps the dimensions
  // and does not change the data.
  reshaped->dims = malloc(shape->n_int64_data * sizeof(int64_t));

  // Note that the dimension that is applied is encoded as a
  // int64 field. So shape its assumed to have data_type int64

  for (int i = 0; i < shape->n_int64_data; i++)
  {
    // The dimension can be n, 0 or -1.
    if (shape->int64_data[i] == 0)
    {
      // If 0 the dimension is not changed
      reshaped->dims[i] = data->dims[i];
    }
    else if (shape->int64_data[i] == -1)
    {
      // If -1 the dimension is inferred from the remaining dim
      // Only 1 parameter can be -1

      // This is ugly af, just to make it work for now
      uint64_t totalDimData = 1;
      for (int j = 0; j < data->n_dims; j++)
      {
        totalDimData *= data->dims[j];
      }

      uint64_t totalShape = 1;

      for (int j = 0; j < shape->n_int64_data; j++)
      {
        if (shape->int64_data[j] > 0)
        {
          totalShape *= shape->int64_data[j];
        }
        else if (shape->int64_data[j] == 0)
        {
          totalShape *= data->dims[j];
        }
        // Just ignore if -1
      }
      reshaped->dims[i] = totalDimData/totalShape;
    }
    else
    {
      reshaped->dims[i] = shape->int64_data[i];
    }
  }

  // Populate some parameters
  reshaped->n_dims       = shape->n_int64_data;
  reshaped->has_raw_data = 0;
  //reshaped->data_type    = data->data_type;
  reshaped->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  reshaped->n_float_data = data->n_float_data;
  reshaped->float_data   = malloc(data->n_float_data * sizeof(float));

  for (int i = 0; i < data->n_float_data; i++) {
    reshaped->float_data[i] = data->float_data[i];
  }

  TRACE_TENSOR(2, true, reshaped);
  TRACE_EXIT(1);
  return 0;
}

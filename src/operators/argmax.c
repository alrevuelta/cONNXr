#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "argmax.h"

void operators_argmax(Onnx__TensorProto *data, int axis, int keepdims, Onnx__TensorProto *reduced)
{
  DEBUG_PRINT("Calling operator_argmax");

  // TODO axis and keepdims is not implemented
  // TODO Only a simple case with 2x2 matrix is implemented

  // TODO in all operators, init unused fields.
  // i.e. if the tensor is int64, init n_float_data to 0

  // Allocte memory
  reduced->dims = malloc(data->dims[1] * sizeof(int64_t));

  // Populate some parameters
  reduced->name         = "name_is_set_afterwards\0";
  reduced->n_dims       = 1;
  reduced->dims[0]      = data->dims[1];
  reduced->has_raw_data = 0;

  // INT64 is hardcoded by design
  reduced->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
  reduced->n_int64_data = data->dims[1];
  reduced->int64_data = malloc(reduced->n_int64_data * sizeof(int64_t));

  switch(data->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      for (int i = 0; i < data->dims[1]; i++)
      {
        // init maxval to the first elemen
        float maxval = data->float_data[i];
        int64_t maxind = 0;
        // todo start from j = 1?
        for (int j = 0; j < data->dims[0]; j++)
        {
          if (data->float_data[i + data->dims[1] * j] > maxval)
          {
            maxind = j;
            maxval = data->float_data[i + data->dims[1] * j];
          }
        }
        reduced->int64_data[i] = maxind;
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      // TODO
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
    {
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      break;
    default:
      break;
  }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "cast.h"

// TODO paste header form onnx doc
void operator_cast(size_t n_input,
                   Onnx__TensorProto **input,
                   size_t n_attribute,
                   Onnx__AttributeProto **attribute,
                   size_t n_output,
                   Onnx__TensorProto **output)
{
  // TODO Scientific notation is not supported, like 1e-5
  // TODO Only float to int64 conversion is supported

  // TODO temporal
  Onnx__TensorProto *T1 = input[0];
  Onnx__TensorProto *T2 = output[0];
  Onnx__AttributeProto *attr = attribute[0];

  DEBUG_PRINT("Calling operator_cast");
  debug_print_dims(T1->n_dims, T1->dims);

  // todo remove. just to make it work
  int to = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;

  T2->dims = malloc(T1->n_dims * sizeof(int64_t));
  for (int i = 0; i < T1->n_dims; i++)
  {
    T2->dims[i] = T1->dims[i];
  }

  // Populate some parameters
  T2->name         = "name_is_set_afterwards\0";
  T2->n_dims       = T1->n_dims;
  T2->has_raw_data = 0;
  T2->data_type    = to;

  // TODO Set unused parameters to 0?

  switch(T1->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      // Hardcode conversion to int64 or double only
      if (to == ONNX__TENSOR_PROTO__DATA_TYPE__INT64)
      {
        T2->n_int64_data = T1->n_float_data;
        T2->int64_data = malloc(T2->n_int64_data * sizeof(int64_t));
        for (int i = 0; i < T2->n_int64_data; i++)
        {
          T2->int64_data[i] = (int64_t)T1->float_data[i];
        }
      }
      else if (to == ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE)
      {
        T2->n_double_data = T1->n_double_data;
        T2->double_data = malloc(T2->n_double_data * sizeof(double));
        for (int i = 0; i < T2->n_double_data; i++)
        {
          T2->double_data[i] = (double)T1->float_data[i];
        }
      }
      // TODO Conversion from float to other types
    }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
    {
    }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      break;
    default:
      break;
  }

  debug_print_dims(T2->n_dims, T2->dims);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "reshape.h"


// Template example
/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  xx xx
 *  \param[in]  xx xx
 *  \param[out] xx xx
 *  \return     xx
 */
void operator_reshape(Onnx__TensorProto *data, Onnx__TensorProto *shape, Onnx__TensorProto *reshaped)
{
  DEBUG_PRINT("Calling operator_reshape");
  debug_print_dims(data->n_dims, data->dims);

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
    DEBUG_PRINT("-----reshaped->dims[%d] = %lld", i, reshaped->dims[i]);
  }

  // Populate some parameters
  reshaped->name         = "name_is_set_afterwards\0";
  reshaped->n_dims       = shape->n_int64_data;
  reshaped->has_raw_data = 0;
  reshaped->data_type    = data->data_type;

  switch(data->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      reshaped->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      reshaped->n_float_data = data->n_float_data;
      reshaped->float_data = malloc(data->n_float_data * sizeof(float));
      for (int i = 0; i < data->n_float_data; i++) {
        reshaped->float_data[i] = data->float_data[i];
      }
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
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      break;
    default:
      break;
  }
  debug_print_dims(reshaped->n_dims, reshaped->dims);
}

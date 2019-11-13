#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../onnx.pb-c.h"
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
void operator_reshape(Onnx__TensorProto *data, Onnx__TensorShapeProto *shape, Onnx__TensorProto *reshaped)
{
  DEBUG_PRINT("Calling operator_cast");

  reshaped->dims = malloc(data->n_dims * sizeof(int64_t));

  for (int i = 0; i < shape->n_dim; i++)
  {
    reshaped->dims[i] = shape->dim[i]->dim_value;
  }

  // Populate some parameters
  reshaped->name         = "name_is_set_afterwards\0";
  reshaped->n_dims       = data->n_dims;
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
}

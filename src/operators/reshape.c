#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "../trace.h"
#include "operators.h"

/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *  Limitations: There might be some limitations with respect to the official onnx
 *  operator. Write here possible limitations, i.e. if the function doesnt work
 *  with all types, or if it works with a specific number of dimensions only
 *
 *  \param[in]      n_input     Number of inputs of the operator
 *  \param[in]      input       Array of pointers to the inputs of the operator
 *  \param[in]      n_attribute Number of attributes of the operator
 *  \param[in]      attribute   Array of pointers to the attributes of the operator
 *  \param[in]      n_output    Numper of outputs of the operator
 *  \param[in/out]  output      Array of pointer to the outputs of the operators
 *  \return         error       Different than 0 if an error was produced
 */
int operator_reshape(operator__context *context)
{
  TRACE_LEVEL0("Calling operator_reshape\n");

  operator__onnx__reshape__context *sc = (operator__onnx__reshape__context *) context;

  // Not sure about this implementation. It just swaps the dimensions
  // and does not change the data.
  sc->out->reshaped->dims = malloc(sc->in->shape->n_int64_data * sizeof(int64_t));

  // Note that the dimension that is applied is encoded as a
  // int64 field. So shape its assumed to have data_type int64

  for (int i = 0; i < sc->in->shape->n_int64_data; i++)
  {
    // The dimension can be n, 0 or -1.
    if (sc->in->shape->int64_data[i] == 0)
    {
      // If 0 the dimension is not changed
      sc->out->reshaped->dims[i] = sc->in->data->dims[i];
    }
    else if (sc->in->shape->int64_data[i] == -1)
    {
      // If -1 the dimension is inferred from the remaining dim
      // Only 1 parameter can be -1

      // This is ugly af, just to make it work for now
      uint64_t totalDimData = 1;
      for (int j = 0; j < sc->in->data->n_dims; j++)
      {
        totalDimData *= sc->in->data->dims[j];
      }

      uint64_t totalShape = 1;

      for (int j = 0; j < sc->in->shape->n_int64_data; j++)
      {
        if (sc->in->shape->int64_data[j] > 0)
        {
          totalShape *= sc->in->shape->int64_data[j];
        }
        else if (sc->in->shape->int64_data[j] == 0)
        {
          totalShape *= sc->in->data->dims[j];
        }
        // Just ignore if -1
      }
      sc->out->reshaped->dims[i] = totalDimData/totalShape;
    }
    else
    {
      sc->out->reshaped->dims[i] = sc->in->shape->int64_data[i];
    }
    TRACE_LEVEL0("-----reshaped->dims[%d] = %" PRId64 "\n", i, sc->out->reshaped->dims[i]);
  }

  // Populate some parameters
  sc->out->reshaped->name         = "name_is_set_afterwards\0";
  sc->out->reshaped->n_dims       = sc->in->shape->n_int64_data;
  sc->out->reshaped->has_raw_data = 0;
  sc->out->reshaped->data_type    = sc->in->data->data_type;

  switch(sc->in->data->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      sc->out->reshaped->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      sc->out->reshaped->n_float_data = sc->in->data->n_float_data;
      sc->out->reshaped->float_data = malloc(sc->in->data->n_float_data * sizeof(float));
      for (int i = 0; i < sc->in->data->n_float_data; i++) {
        sc->out->reshaped->float_data[i] = sc->in->data->float_data[i];
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
  debug_print_dims(sc->out->reshaped->n_dims, sc->out->reshaped->dims);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"
#include "utils.h"

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
int operator_argmax(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_argmax\n");

  Onnx__TensorProto *axis = searchInputByName(ctx, 0);
  //Onnx__TensorProto *keepdims = searchInputByName(ctx, 1);
  //Onnx__TensorProto *select_last_index = searchInputByName(ctx, 2);

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

  debug_print_dims(axis->n_dims, axis->dims);

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

  switch(axis->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
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
  debug_print_dims(data->n_dims, data->dims);
  return 0;
}

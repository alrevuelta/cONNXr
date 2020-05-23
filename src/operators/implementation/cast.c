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
int operator_cast(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_cast\n");

  Onnx__TensorProto *input = searchInputByName(ctx, 0);

  Onnx__TensorProto *output = searchOutputByName(ctx, 0);

  debug_print_dims(input->n_dims, input->dims);

  if (0){
    // TODO Just a prototype. Not tested
    // TODO Scientific notation is not supported, like 1e-5
    // TODO Only float to int64 conversion is supported
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return -1;
  }

  /* TODO: This is hardcoded */
  int to = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;

  output->dims = malloc(input->n_dims * sizeof(int64_t));
  for (int i = 0; i < input->n_dims; i++)
  {
    output->dims[i] = input->dims[i];
  }

  // Populate some parameters
  output->n_dims       = input->n_dims;
  output->has_raw_data = 0;
  output->data_type    = to;

  // TODO Set unused parameters to 0?
  switch(input->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      // Hardcode conversion to int64 or double only
      if (to == ONNX__TENSOR_PROTO__DATA_TYPE__INT64)
      {
        output->n_int64_data = input->n_float_data;
        output->int64_data = malloc(output->n_int64_data * sizeof(int64_t));
        for (int i = 0; i < output->n_int64_data; i++)
        {
          output->int64_data[i] = (int64_t)input->float_data[i];
        }
      }
      else if (to == ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE)
      {
        output->n_double_data = input->n_double_data;
        output->double_data = malloc(output->n_double_data * sizeof(double));
        for (int i = 0; i < output->n_double_data; i++)
        {
          output->double_data[i] = (double)input->float_data[i];
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

  debug_print_dims(output->n_dims, output->dims);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
int operator_reshape(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_reshape\n");

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  debug_print_dims(input[0]->n_dims, input[0]->dims);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return -1;
  }

  // Not sure about this implementation. It just swaps the dimensions
  // and does not change the data.
  output[0]->dims = malloc(input[1]->n_int64_data * sizeof(int64_t));

  // Note that the dimension that is applied is encoded as a
  // int64 field. So shape its assumed to have data_type int64

  for (int i = 0; i < input[1]->n_int64_data; i++)
  {
    // The dimension can be n, 0 or -1.
    if (input[1]->int64_data[i] == 0)
    {
      // If 0 the dimension is not changed
      output[0]->dims[i] = input[0]->dims[i];
    }
    else if (input[1]->int64_data[i] == -1)
    {
      // If -1 the dimension is inferred from the remaining dim
      // Only 1 parameter can be -1

      // This is ugly af, just to make it work for now
      uint64_t totalDimData = 1;
      for (int j = 0; j < input[0]->n_dims; j++)
      {
        totalDimData *= input[0]->dims[j];
      }

      uint64_t totalShape = 1;

      for (int j = 0; j < input[1]->n_int64_data; j++)
      {
        if (input[1]->int64_data[j] > 0)
        {
          totalShape *= input[1]->int64_data[j];
        }
        else if (input[1]->int64_data[j] == 0)
        {
          totalShape *= input[0]->dims[j];
        }
        // Just ignore if -1
      }
      output[0]->dims[i] = totalDimData/totalShape;
    }
    else
    {
      output[0]->dims[i] = input[1]->int64_data[i];
    }
    TRACE_LEVEL0("-----reshaped->dims[%d] = %lld\n", i, output[0]->dims[i]);
  }

  // Populate some parameters
  output[0]->name         = "name_is_set_afterwards\0";
  output[0]->n_dims       = input[1]->n_int64_data;
  output[0]->has_raw_data = 0;
  output[0]->data_type    = input[0]->data_type;

  switch(input[0]->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->n_float_data = input[0]->n_float_data;
      output[0]->float_data = malloc(input[0]->n_float_data * sizeof(float));
      for (int i = 0; i < input[0]->n_float_data; i++) {
        output[0]->float_data[i] = input[0]->float_data[i];
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
  debug_print_dims(output[0]->n_dims, output[0]->dims);
  return 0;
}

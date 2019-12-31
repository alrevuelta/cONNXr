#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "matmul.h"

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
int operator_matmul(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_matmul");
  debug_print_dims(input[0]->n_dims, input[0]->dims);
  debug_print_dims(input[1]->n_dims, input[1]->dims);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return -1;
  }

  // TODO Hardcoded for 2 dimensions
  // TODO Might be useful to define a macro like
  // #define I(a,b,c,d) I[(a)+(b)*oH+(c)*oH*oW+(d)*oH*oW*C]
  // dont know how to handle the different dimensions though

  // Allocte memory
  output[0]->dims = malloc(2 * sizeof(int64_t));

  // Populate some parameters
  // TODO: Is this working? No mem is allocated
  output[0]->name         = "name_is_set_afterwards\0";
  output[0]->n_dims       = 2;
  output[0]->dims[0]      = input[0]->dims[0];
  output[0]->dims[1]      = input[1]->dims[1];
  output[0]->has_raw_data = 0;

  switch(input[0]->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->n_float_data = input[0]->dims[0] * input[1]->dims[1];
      output[0]->float_data = malloc(input[0]->dims[0] * input[1]->dims[1] * sizeof(float));
      for (int i = 0; i < input[0]->dims[0]; i++) {
        for (int j = 0; j < input[1]->dims[1]; j++) {
          float sum = 0;
          for (int p = 0; p < input[0]->dims[1]; p++) {
            sum += (input[0]->float_data[i*input[0]->dims[1]+p] * input[1]->float_data[p*input[1]->dims[1]+j]);
            // Saturate the value?
          }
          output[0]->float_data[i*input[1]->dims[1]+j] = sum;
        }
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT32;
      output[0]->n_int32_data = input[0]->dims[0] * input[1]->dims[1];
      output[0]->int32_data = malloc(input[0]->dims[0] * input[1]->dims[1] * sizeof(int32_t));
      for (int i = 0; i < input[0]->dims[0]; i++) {
        for (int j = 0; j < input[1]->dims[1]; j++) {
          int32_t sum = 0;
          for (int p = 0; p < input[0]->dims[1]; p++) {
            sum += (input[0]->int32_data[i*input[0]->dims[1]+p] * input[1]->int32_data[p*input[1]->dims[1]+j]);
            // Saturate the value?
          }
          output[0]->int32_data[i*input[1]->dims[1]+j] = sum;
        }
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
    {
      // TODO
      // Use n_int64_data, int64_data
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
    {
      // TODO
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
    {
      // TODO
      // Use n_double_data, double_data
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
    {
      // TODO
      // Note sure but use n_uint64_data and uint64_data (same as uint64)
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
    {
      // TODO
      // Note sure but use n_uint64_data and uint64_data (same as uint64)
    } break;
    default:
      break;
  }

  debug_print_dims(output[0]->n_dims, output[0]->dims);
  return 0;
}

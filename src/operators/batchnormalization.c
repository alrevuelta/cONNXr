#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "batchnormalization.h"

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
int operator_batchnormalization(size_t n_input,
                                Onnx__TensorProto **input,
                                size_t n_attribute,
                                Onnx__AttributeProto **attribute,
                                size_t n_output,
                                Onnx__TensorProto **output)
{
  DEBUG_PRINT("Calling operator_batchnormalization");

  //Onnx__AttributeProto *momentumAttr = searchAttributeNyName(n_attribute, attribute, "momentum");
  debug_print_attributes(n_attribute, attribute);
  /* Epsilon is hardcoded to float */
  float eps = 0.00001; /* Default value */

  if (n_attribute == 1){
    eps = attribute[0]->f;
  }

  // Allocte memory
  output[0]->dims = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims = input[0]->n_dims;
  for(int i = 0; i < output[0]->n_dims; i++){
    output[0]->dims[i] = input[0]->dims[i];
  }

  // Populate some parameters
  // TODO: Is this working? No mem is allocated
  output[0]->name         = "name_is_set_afterwards\0";
  output[0]->has_raw_data = 0;

  /* The order of inputs is assumed to be:
   * [0] X
   * [1] scale
   * [2] B
   * [3] mean
   * [4] var
   */
  switch(input[0]->data_type)
  {
    /* TODO Hardcoded for N x C x D1 x D2 */
    /* Input tensors have dimension C */
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->n_float_data = input[0]->n_float_data;
      output[0]->float_data = malloc(output[0]->n_float_data * sizeof(float));
      for (int i = 0; i < output[0]->n_float_data; i++) {
        int index = (i/(input[0]->dims[2] * input[0]->dims[3])) % input[0]->dims[1];
        float mean = input[3]->float_data[index];
        float var = input[4]->float_data[index];
        float scale = input[1]->float_data[index];
        float B = input[2]->float_data[index];
        //printf("input=%f\n", input[0]->float_data[i]);
        //printf("index=%dmean=%f, var=%f, scale=%f, B=%f\n", index, mean, var, scale, B);
        output[0]->float_data[i] = (input[0]->float_data[i] - mean) / sqrtf(var + eps);
        output[0]->float_data[i] = scale * output[0]->float_data[i] + B;
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
    {
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

  return 0;
}

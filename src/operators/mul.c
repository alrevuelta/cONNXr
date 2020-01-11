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
 int operator_mul(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_mul\n");

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  /* TODO: Hardcoded for tiny YOLO */

  /* Move this block to a common function */
  output[0]->dims   = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims = input[0]->n_dims;

  for (int i = 0; i < input[0]->n_dims; i++)
  {
    output[0]->dims[i] = input[0]->dims[i];
  }
  output[0]->has_raw_data = 0;
  output[0]->data_type = input[0]->data_type;

  output[0]->n_float_data = input[0]->n_float_data;
  output[0]->float_data = malloc(output[0]->n_float_data * sizeof(float));

  for (int i = 0; i < input[0]->n_float_data; i++){
    output[0]->float_data[i] = input[0]->float_data[i] * input[1]->float_data[0];
  }
  return 0;
}

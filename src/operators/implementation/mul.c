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
 int operator_mul(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_mul\n");

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);

  Onnx__TensorProto *C = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  /* TODO: Hardcoded for tiny YOLO */

  /* Move this block to a common function */
  C->dims   = malloc(A->n_dims * sizeof(int64_t));
  C->n_dims = A->n_dims;

  for (int i = 0; i < A->n_dims; i++)
  {
    C->dims[i] = A->dims[i];
  }
  C->has_raw_data = 0;
  C->data_type = A->data_type;

  C->n_float_data = A->n_float_data;
  C->float_data = malloc(C->n_float_data * sizeof(float));

  for (int i = 0; i < A->n_float_data; i++){
    C->float_data[i] = A->float_data[i] * B->float_data[0];
  }
  return 0;
}

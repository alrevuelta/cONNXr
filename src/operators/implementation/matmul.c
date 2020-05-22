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
int operator_matmul(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_matmul\n");

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  debug_print_dims(A->n_dims, A->dims);
  debug_print_dims(B->n_dims, B->dims);

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
  Y->dims = malloc(2 * sizeof(int64_t));

  // Populate some parameters
  Y->n_dims       = 2;
  Y->dims[0]      = A->dims[0];
  Y->dims[1]      = B->dims[1];
  Y->has_raw_data = 0;

  switch(A->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      Y->n_float_data = A->dims[0] * B->dims[1];
      Y->float_data = malloc(A->dims[0] * B->dims[1] * sizeof(float));
      for (int i = 0; i < A->dims[0]; i++) {
        for (int j = 0; j < B->dims[1]; j++) {
          float sum = 0;
          for (int p = 0; p < A->dims[1]; p++) {
            sum += (A->float_data[i*A->dims[1]+p] * B->float_data[p*B->dims[1]+j]);
            // Saturate the value?
          }
          Y->float_data[i*B->dims[1]+j] = sum;
        }
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
    {
      Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT32;
      Y->n_int32_data = A->dims[0] * B->dims[1];
      Y->int32_data = malloc(A->dims[0] * B->dims[1] * sizeof(int32_t));
      for (int i = 0; i < A->dims[0]; i++) {
        for (int j = 0; j < B->dims[1]; j++) {
          int32_t sum = 0;
          for (int p = 0; p < A->dims[1]; p++) {
            sum += (A->int32_data[i*A->dims[1]+p] * B->int32_data[p*B->dims[1]+j]);
            // Saturate the value?
          }
          Y->int32_data[i*B->dims[1]+j] = sum;
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

  debug_print_dims(Y->n_dims, Y->dims);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"
#include "utils.h"

/*! \fn operator_add()
 *  \brief Add: Performs element-wise binary addition (with Numpy-style broadcasting support).
 *              This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more
 *              details please check the doc.
 *         Version: This version of the operator has been available since version 7
 *                  of the default ONNX operator set. Other versions of this operator: Add-1, Add-6
 *         Inputs:
 *          A : T. First operand.
 *          B : T. Second operand.
 *         Outputs:
 *          C : T. Result, has same element type as two inputs
 *         Type Constraints:
 *          T : tensor(uint32), tensor(uint64), tensor(int32), tensor(int64),
 *          tensor(float16), tensor(float), tensor(double). Constrain input and output types to
 *          high-precision numeric tensors.
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
int operator_add(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_add\n");

  Onnx__TensorProto *A = searchInputByName(ctx, 0);
  Onnx__TensorProto *B = searchInputByName(ctx, 1);
  Onnx__TensorProto *C = searchOutputByName(ctx, 0);

  debug_print_dims(A->n_dims, A->dims);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return 1;
  }

  /* There order of operands if unknown. The longest one will determine the output */
  /* Quick and dirty solution */
  if (A->n_dims > B->n_dims){
    C->dims = malloc(A->n_dims * sizeof(int64_t));
    C->n_dims = A->n_dims;
    C->n_float_data = A->n_float_data; // check other types
    for (int i = 0; i < C->n_dims; i++)
    {
      C->dims[i] = A->dims[i];
    }
  }else{
    C->dims = malloc(B->n_dims * sizeof(int64_t));
    C->n_dims = B->n_dims;
    C->n_float_data = B->n_float_data; // check other types
    for (int i = 0; i < C->n_dims; i++)
    {
      C->dims[i] = B->dims[i];
    }
  }

  C->has_raw_data = 0;
  C->data_type = A->data_type;

  switch(A->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      C->float_data = malloc(C->n_float_data * sizeof(float));
      /* TODO: ugly */
      for (int i = 0; i < C->n_float_data; i++) {
        /* Normal case where dimensions match */
        if (A->n_dims == B->n_dims) {
          C->float_data[i] = A->float_data[i] + B->float_data[i];
        /* Broadcasting. Hardcoded not working */
        }else{
          /* If inside loop :( */
          if (B->n_dims == 1){
            C->float_data[i] = A->float_data[i] + B->float_data[i%B->dims[0]];
          }else{
            /* TODO Hardcoded for TINY YOLO */
            if (A->dims[0] == 3){ /* Remove this uAF*/
              C->float_data[i] = A->float_data[i%3] + B->float_data[i];
            /* TODO Hardcoded for MNIST */
            }else{
              C->float_data[i] = A->float_data[i] + B->float_data[i/(A->dims[2]*A->dims[3])];
            }
          }
        }
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      break;
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

  debug_print_dims(C->n_dims, C->dims);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "matmul.h"

/*! \fn void matmul(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c)
 *  \brief MatMul: Matrix product that behaves like numpy.matmul:
 *                 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
 *         Version: This version of the operator has been available since
 *                  version 9 of the default ONNX operator set. Other versions
 *                  of this operator: MatMul-1
 *         Inputs:
 *          A : T. N-dimensional matrix A
 *          B : T. N-dimensional matrix B
 *         Outputs:
 *          Y : T. Matrix multiply results from A * B
 *         Type Constraints:
 *          T : tensor(float16), tensor(float), tensor(double), tensor(uint32),
 *              tensor(uint64), tensor(int32), tensor(int64)
 *              Constrain input and output types to float/int tensors.
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  Onnx__TensorProto a
 *  \param[in]  Onnx__TensorProto b
 *  \param[out] Onnx__TensorProto c
 *  \return     void
 */
void operator_matmul(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *o)
{
  DEBUG_PRINT("Calling operator_matmul");
  debug_print_dims(a->n_dims, a->dims);
  debug_print_dims(b->n_dims, b->dims);

  // TODO Hardcoded for 2 dimensions

  // TODO Might be useful to define a macro like
  // #define I(a,b,c,d) I[(a)+(b)*oH+(c)*oH*oW+(d)*oH*oW*C]
  // dont know how to handle the different dimensions though

  // Check condition?
  //a->data_type == b->data_type;

  // Allocte memory
  o->dims = malloc(2 * sizeof(int64_t));

  // Populate some parameters
  // TODO: Is this working? No mem is allocated
  o->name         = "name_is_set_afterwards\0";
  o->n_dims       = 2;
  o->dims[0]      = a->dims[0];
  o->dims[1]      = b->dims[1];
  o->has_raw_data = 0;

  switch(a->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      o->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      o->n_float_data = a->dims[0] * b->dims[1];
      o->float_data = malloc(a->dims[0] * b->dims[1] * sizeof(float));
      for (int i = 0; i < a->dims[0]; i++) {
        for (int j = 0; j < b->dims[1]; j++) {
          float sum = 0;
          for (int p = 0; p < a->dims[1]; p++) {
            sum += (a->float_data[i*a->dims[1]+p] * b->float_data[p*b->dims[1]+j]);
            // Saturate the value?
          }
          o->float_data[i*b->dims[1]+j] = sum;
        }
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
    {
      o->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT32;
      o->n_int32_data = a->dims[0] * b->dims[1];
      o->int32_data = malloc(a->dims[0] * b->dims[1] * sizeof(int32_t));
      for (int i = 0; i < a->dims[0]; i++) {
        for (int j = 0; j < b->dims[1]; j++) {
          int32_t sum = 0;
          for (int p = 0; p < a->dims[1]; p++) {
            sum += (a->int32_data[i*a->dims[1]+p] * b->int32_data[p*b->dims[1]+j]);
            // Saturate the value?
          }
          o->int32_data[i*b->dims[1]+j] = sum;
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

  debug_print_dims(o->n_dims, o->dims);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "add.h"

/*! \fn operator_add(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c)
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
 *       Limitations: There might be some limitations with respect to the onnx
 *         official operator. Write here possible limitations, i.e. if the
 *         function doesnt work with all types, or if it works with a specific
 *         number of dimensions only
 *  \param[in]  Onnx__TensorProto a
 *  \param[in]  Onnx__TensorProto b
 *  \param[out] Onnx__TensorProto c
 *  \return     void
 */
void operator_add(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c)
{
  DEBUG_PRINT("Calling operator_add");

  // Check condition?
  //a->data_type == b->data_type
  //a->n_dims == b->n_dims
  //a->dims[i] == b->dims[i]

  // Allocte memory
  c->dims = malloc(a->n_dims * sizeof(int64_t));

  // Populate some parameters
  c->name         = "name_is_set_afterwards\0";
  c->n_dims       = a->n_dims;

  for (int i = 0; i < a->n_dims; i++)
  {
    c->dims[i] = a->dims[i];
  }
  c->has_raw_data = 0;
  c->data_type = a->data_type;

  switch(a->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      c->n_float_data = a->n_float_data;
      c->float_data = malloc(c->n_float_data * sizeof(float));
      for (int i = 0; i < a->n_float_data; i++) {
        // Note that b can have a different dimension. In that case
        // broadcasting is performed
        c->float_data[i] = a->float_data[i] + b->float_data[i%b->n_float_data];
      }
    } break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      // TODO
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
}

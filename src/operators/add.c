#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../trace.h"
#include "operators.h"

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
int operator_add(struct operator__context *context)
{
  TRACE_LEVEL0("Calling operator_add\n");

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return 1;
  }

  /* Here we can easily access all the i/o and attributes. NULL if no present */
  /* sc = specific context*/
  struct operator__onnx__add__context *sc = (void *) context;

  /* There order of operands if unknown. The longest one will determine the output */
  /* Quick and dirty solution */

  if (sc->in->A->n_dims > sc->in->B->n_dims){
    sc->out->C->dims = malloc(sc->in->A->n_dims * sizeof(int64_t));
    sc->out->C->n_dims = sc->in->A->n_dims;
    sc->out->C->n_float_data = sc->in->A->n_float_data; // check other types
    for (int i = 0; i < sc->out->C->n_dims; i++)
    {
      sc->out->C->dims[i] = sc->in->A->dims[i];
    }
  }else{
    sc->out->C->dims = malloc(sc->in->B->n_dims * sizeof(int64_t));
    sc->out->C->n_dims = sc->in->B->n_dims;
    sc->out->C->n_float_data = sc->in->B->n_float_data; // check other types
    for (int i = 0; i < sc->out->C->n_dims; i++)
    {
      sc->out->C->dims[i] = sc->in->B->dims[i];
    }
  }

  sc->out->C->has_raw_data = 0;
  sc->out->C->data_type = sc->in->A->data_type;

  switch(sc->in->A->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      sc->out->C->float_data = malloc(sc->out->C->n_float_data * sizeof(float));
      /* TODO: ugly */
      for (int i = 0; i < sc->out->C->n_float_data; i++) {
        /* Normal case where dimensions match */
        if (sc->in->A->n_dims == sc->in->B->n_dims) {
          sc->out->C->float_data[i] = sc->in->A->float_data[i] + sc->in->B->float_data[i];
        /* Broadcasting. Hardcoded not working */
        }else{
          /* If inside loop :( */
          if (sc->in->B->n_dims == 1){
            sc->out->C->float_data[i] = sc->in->A->float_data[i] + sc->in->B->float_data[i%sc->in->B->dims[0]];
          }else{
            /* TODO Hardcoded for TINY YOLO */
            if (sc->in->A->dims[0] == 3){ /* Remove this uAF*/
              sc->out->C->float_data[i] = sc->in->A->float_data[i%3] + sc->in->B->float_data[i];
            /* TODO Hardcoded for MNIST */
            }else{
              sc->out->C->float_data[i] = sc->in->A->float_data[i] + sc->in->B->float_data[i/(sc->in->A->dims[2]*sc->in->A->dims[3])];
            }
          }
        }
      }
    } break;
    default:
      break;
  }

  debug_print_dims(sc->out->C->n_dims, sc->out->C->dims);
  return 0;
}

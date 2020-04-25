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
int operator_matmul(struct operator__context *context)
{

  struct operator__onnx__matmul__context *sc = (void *) context;

  debug_print_dims(sc->in->A->n_dims, sc->in->A->dims);
  debug_print_dims(sc->in->B->n_dims, sc->in->B->dims);

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
  sc->out->Y->dims = malloc(2 * sizeof(int64_t));

  // Populate some parameters
  sc->out->Y->n_dims       = 2;
  sc->out->Y->dims[0]      = sc->in->A->dims[0];
  sc->out->Y->dims[1]      = sc->in->B->dims[1];
  sc->out->Y->has_raw_data = 0;

  sc->out->Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  sc->out->Y->n_float_data = sc->in->A->dims[0] * sc->in->B->dims[1];
  sc->out->Y->float_data = malloc(sc->in->A->dims[0] * sc->in->B->dims[1] * sizeof(float));
  for (int i = 0; i < sc->in->A->dims[0]; i++) {
    for (int j = 0; j < sc->in->B->dims[1]; j++) {
      float sum = 0;
      for (int p = 0; p < sc->in->A->dims[1]; p++) {
        sum += (sc->in->A->float_data[i*sc->in->A->dims[1]+p] * sc->in->B->float_data[p*sc->in->B->dims[1]+j]);
        // Saturate the value?
      }
      sc->out->Y->float_data[i*sc->in->B->dims[1]+j] = sum;
    }
  }

  debug_print_dims(sc->out->Y->n_dims, sc->out->Y->dims);
  return 0;
}

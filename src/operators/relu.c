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
int operator_relu(struct operator__context *context)
 {
   TRACE_LEVEL0("Calling operator_relu\n");

   struct operator__onnx__relu__context *sc = (void *) context;

   debug_print_dims(sc->in->X->n_dims, sc->in->X->dims);

   sc->out->Y->dims = malloc(sc->in->X->n_dims * sizeof(int64_t));
   for (int i = 0; i < sc->in->X->n_dims; i++)
   {
     sc->out->Y->dims[i] = sc->in->X->dims[i];
   }

   // Populate some parameters
   sc->out->Y->n_dims       = sc->in->X->n_dims;
   sc->out->Y->has_raw_data = 0;
   sc->out->Y->data_type    = sc->in->X->data_type;

   sc->out->Y->n_float_data = sc->in->X->n_float_data;
   sc->out->Y->float_data = malloc(sc->out->Y->n_float_data * sizeof(float));
   for (int i = 0; i < sc->out->Y->n_float_data; i++)
   {
     sc->out->Y->float_data[i] = sc->in->X->float_data[i] < 0 ? 0 : sc->in->X->float_data[i];
   }

   debug_print_dims(sc->out->Y->n_dims, sc->out->Y->dims);
   return 0;
 }

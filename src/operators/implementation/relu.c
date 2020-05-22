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
int operator_relu(node_context *ctx)
 {
   TRACE_LEVEL0("Calling operator_relu\n");

   Onnx__TensorProto *X = searchInputByName(ctx, 0);
   
   Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

   debug_print_dims(X->n_dims, X->dims);

   if (0){
     /* TODO: Check some conditions. For example if a specific
      * functionality is not supported */
     //a->data_type == b->data_type
     //a->n_dims == b->n_dims
     //a->dims[i] == b->dims[i]
     return 1;
   }

   Y->dims = malloc(X->n_dims * sizeof(int64_t));
   for (int i = 0; i < X->n_dims; i++)
   {
     Y->dims[i] = X->dims[i];
   }

   // Populate some parameters
   Y->n_dims       = X->n_dims;
   Y->has_raw_data = 0;
   Y->data_type    = X->data_type;

   switch(X->data_type)
   {
     case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
     {
       Y->n_float_data = X->n_float_data;
       Y->float_data = malloc(Y->n_float_data * sizeof(float));
       for (int i = 0; i < Y->n_float_data; i++)
       {
         Y->float_data[i] = X->float_data[i] < 0 ? 0 : X->float_data[i];
       }
     }
       break;
     case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
     {
       // TODO
     }
       break;
     case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
     {
       Y->n_double_data = X->n_double_data;
       Y->double_data = malloc(Y->n_double_data * sizeof(double));
       for (int i = 0; i < Y->n_double_data; i++)
       {
         Y->double_data[i] = X->double_data[i] < 0 ? 0 : X->double_data[i];
       }
     }
       break;
     default:
       break;
   }
   debug_print_dims(Y->n_dims, Y->dims);
   return 0;
 }

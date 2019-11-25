#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "relu.h"

// Template example
/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  xx xx
 *  \param[in]  xx xx
 *  \param[out] xx xx
 *  \return     xx
 */
 void operator_relu(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output)
 {
   DEBUG_PRINT("Calling operator_relu");

   // TODO temporal
   Onnx__TensorProto *X = input[0];
   Onnx__TensorProto *Y = output[0];

   debug_print_dims(X->n_dims, X->dims);

   Y->dims = malloc(X->n_dims * sizeof(int64_t));
   for (int i = 0; i < X->n_dims; i++)
   {
     Y->dims[i] = X->dims[i];
   }



   // Populate some parameters
   Y->name         = "name_is_set_afterwards\0";
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
 }

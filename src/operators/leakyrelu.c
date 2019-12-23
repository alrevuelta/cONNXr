#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "leakyrelu.h"

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
 int operator_leakyrelu(size_t n_input,
                        Onnx__TensorProto **input,
                        size_t n_attribute,
                        Onnx__AttributeProto **attribute,
                        size_t n_output,
                        Onnx__TensorProto **output)
 {
   DEBUG_PRINT("Calling operator_leakyrelu");
   debug_print_dims(input[0]->n_dims, input[0]->dims);

   /* TODO Alpha is not always float */
   /* Default value */
   float alpha = 0.01;

   /* If there is an attribute, its alpha */
   if (n_attribute == 1){
     alpha = attribute[0]->f;
   }

   if (0){
     /* TODO: Check some conditions. For example if a specific
      * functionality is not supported */
     //a->data_type == b->data_type
     //a->n_dims == b->n_dims
     //a->dims[i] == b->dims[i]
     return -1;
   }

   output[0]->dims = malloc(input[0]->n_dims * sizeof(int64_t));
   for (int i = 0; i < input[0]->n_dims; i++)
   {
     output[0]->dims[i] = input[0]->dims[i];
   }

   // Populate some parameters
   output[0]->name         = "name_is_set_afterwards\0";
   output[0]->n_dims       = input[0]->n_dims;
   output[0]->has_raw_data = 0;
   output[0]->data_type    = input[0]->data_type;

   switch(input[0]->data_type)
   {
     case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
     {
       output[0]->n_float_data = input[0]->n_float_data;
       output[0]->float_data = malloc(output[0]->n_float_data * sizeof(float));
       for (int i = 0; i < output[0]->n_float_data; i++)
       {
         output[0]->float_data[i] = input[0]->float_data[i] < 0 ?
                                    input[0]->float_data[i] * alpha :
                                    input[0]->float_data[i];
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
       output[0]->n_double_data = input[0]->n_double_data;
       output[0]->double_data = malloc(output[0]->n_double_data * sizeof(double));
       for (int i = 0; i < output[0]->n_double_data; i++)
       {
         output[0]->double_data[i] = input[0]->double_data[i] < 0 ? 0 : input[0]->double_data[i];
       }
     }
       break;
     default:
       break;
   }
   debug_print_dims(output[0]->n_dims, output[0]->dims);
   return 0;
 }

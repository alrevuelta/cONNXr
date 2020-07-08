#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

 operator_status operator__onnx__leakyrelu__6__T_tensor_float(
     node_context *ctx
 )
 {

   Onnx__TensorProto *X = searchInputByName(ctx, 0);

   Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

   TRACE_LEVEL0("Calling operator_leakyrelu\n");

   if (0){
     /* TODO: Check some conditions. For example if a specific
      * functionality is not supported */
     return -1;
   }

   debug_print_dims(X->n_dims, X->dims);

   /* TODO Alpha is not always float */
   /* Default value */
   float alpha = 0.01;

   /* If there is an attribute, its alpha */
   if (ctx->onnx_node->n_attribute == 1){
     alpha = ctx->onnx_node->attribute[0]->f;
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

   Y->n_float_data = X->n_float_data;
   Y->float_data = malloc(Y->n_float_data * sizeof(float));
   for (int i = 0; i < Y->n_float_data; i++)
   {
     Y->float_data[i] = X->float_data[i] < 0 ?
                                X->float_data[i] * alpha :
                                X->float_data[i];
   }

   debug_print_dims(Y->n_dims, Y->dims);
   return 0;
 }

/* This is not being used. Something might be reused */
#if 0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"
#include "utils.h"

/* TODO not very nice, rethink this. Round to even:
https://en.wikipedia.org/wiki/Rounding
*/
int32_t divAndRoundEven(float a, float b)
{
  int32_t AdivB = (int32_t)(a/b);

  /* TODO: AdivB has to be rounded to even! Not implemented*/

  return AdivB;
}

int operator_quantizelinear(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_quantizelinear\n");

  Onnx__TensorProto *X = searchInputByName(ctx, 0);
  Onnx__TensorProto *y_scale = searchInputByName(ctx, 1);
  Onnx__TensorProto *y_zero_point = searchInputByName(ctx, 2);

  Onnx__TensorProto *y = searchOutputByName(ctx, 0);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  y->dims   = malloc(X->n_dims * sizeof(int64_t));
  y->n_dims = X->n_dims;

  for (int i = 0; i < y->n_dims; i++)
  {
    y->dims[i] = X->dims[i];
  }
  y->has_raw_data = 0;

  printf("%f\n", y_scale->float_data[0]);

  printf("%d\n", y_zero_point->int32_data[0]);

  /* TODO hardcoded to uint8
   [0, 255] if it's uint8, or [-128, 127] if it's int8.*/
  y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UINT8;

  y->n_int32_data = X->n_float_data;
  y->int32_data = malloc(y->n_int32_data * sizeof(int32_t));
  /* TODO Only FLOAT is handled*/
  printf("type = %d\n", X->data_type);
  if (X->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT){
    /* TODO third parameter is options, its assumed its always there */
    if (ctx->onnx_node->n_input != 3){return 1;}
    for(int i = 0; i < y->n_int32_data; i++){
      int32_t value = divAndRoundEven(X->float_data[i], y_scale->float_data[0]);/* +
                      y_zero_point->int32_data[0];*/
      /* TODO Quick implementation. Find a better way to saturate and avoid negative */
      value > 255 ? value = 255 : value;
      value < 0   ? value = 0   : value;
      y->int32_data[i] = value;
    }
  }else{
    printf("wrong type %d\n", X->data_type);
    return 1;
  }

  return 0;
}
#endif

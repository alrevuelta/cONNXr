#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"

/* TODO not very nice, rethink this. Round to even:
https://en.wikipedia.org/wiki/Rounding
*/
int32_t divAndRoundEven(float a, float b)
{
  int32_t AdivB = (int32_t)(a/b);

  /* TODO: AdivB has to be rounded to even! Not implemented*/

  return AdivB;
}

int operator_quantizelinear(size_t n_input,
                            Onnx__TensorProto **input,
                            size_t n_attribute,
                            Onnx__AttributeProto **attribute,
                            size_t n_output,
                            Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_quantizelinear\n");
  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  output[0]->dims   = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims = input[0]->n_dims;

  for (int i = 0; i < output[0]->n_dims; i++)
  {
    output[0]->dims[i] = input[0]->dims[i];
  }
  output[0]->has_raw_data = 0;

  printf("%f\n", input[1]->float_data[0]);

  printf("%d\n", input[2]->int32_data[0]);

  /* TODO hardcoded to uint8
   [0, 255] if it's uint8, or [-128, 127] if it's int8.*/
  output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UINT8;

  output[0]->n_int32_data = input[0]->n_float_data;
  output[0]->int32_data = malloc(output[0]->n_int32_data * sizeof(int32_t));
  /* TODO Only FLOAT is handled*/
  printf("type = %d\n", input[0]->data_type);
  if (input[0]->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT){
    /* TODO third parameter is options, its assumed its always there */
    if (n_input != 3){return 1;}
    for(int i = 0; i < output[0]->n_int32_data; i++){
      int32_t value = divAndRoundEven(input[0]->float_data[i], input[1]->float_data[0]);/* +
                      input[2]->int32_data[0];*/
      /* TODO Quick implementation. Find a better way to saturate and avoid negative */
      value > 255 ? value = 255 : value;
      value < 0   ? value = 0   : value;
      output[0]->int32_data[i] = value;
    }
  }else{
    printf("wrong type %d\n", input[0]->data_type);
    return 1;
  }

  return 0;
}

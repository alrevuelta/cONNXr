#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

operator_status operator__onnx__transpose__1__T_tensor_float(
    node_context *ctx
)
{
  TRACE_LEVEL0("Calling operator_transpose\n");

  Onnx__TensorProto *data = searchInputByName(ctx, 0);
  Onnx__TensorProto *transposed = searchOutputByName(ctx, 0);
  Onnx__AttributeProto *perm = searchAttributeNyName(ctx->onnx_node->n_attribute,
                                                     ctx->onnx_node->attribute,
                                                     "perm");

  printf("first attribute %zu\n", ctx->onnx_node->n_attribute);
  printf("name value = %s\n", perm->name);
  printf("n_tensors %zu\n", perm->n_tensors);
  printf("type %u\n", perm->type);

  transposed->n_dims = perm->n_ints;
  transposed->dims   = malloc(transposed->n_dims * sizeof(int64_t));

  transposed->has_raw_data = 0;
  transposed->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  transposed->n_float_data = data->n_float_data;

  transposed->float_data   = malloc(transposed->n_float_data * sizeof(float));
  //transposed->float_data = data->float_data;

  // This should be the only case
  if (perm->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS){
    /*size_t n_ints;
    int64_t *ints;*/
    for (size_t i = 0; i < perm->n_ints; i++){
      printf("[%zu]=%lld\n", i, perm->ints[i]);
      transposed->dims[i] = data->dims[perm->ints[i]];
    }
  }

  /*3d case*/
  for (size_t i = 0; i < transposed->dims[0]; i++){
    for (size_t j = 0; j < transposed->dims[1]; j++){
      for (size_t k = 0; k < transposed->dims[2]; k++){
        uint64_t out_index = k + transposed->dims[2]*j + transposed->dims[2]*transposed->dims[1]*i;

        // this kind of works
        //uint64_t new_index = j + data->dims[2]*k + data->dims[2]*data->dims[1]*i;
        uint64_t new_index = k + data->dims[2]*j + data->dims[2]*data->dims[1]*i;
        /* */
        printf("[%zu][%zu][%zu] = [%zu][%zu][%zu] out=%lld, new=%lld\n", i, j, k, i, k, j, out_index, new_index);
        transposed->float_data[out_index] = data->float_data[new_index];
        }
      }
    }
    /*input index*/
    /*
    for (int64_t i = 0; i < data->n_float_data; i++){
      size_t x = i/(data->dims[1] * data->dims[2]);
      size_t y = (i%(data->dims[1]*data->dims[2]))/data->dims[2];
      size_t z = (i%(data->dims[1]*data->dims[2]))%data->dims[2];

      size_t xx = i/(transposed->dims[1] * transposed->dims[2]);
      size_t yy = (i%(transposed->dims[1]*transposed->dims[2]))/transposed->dims[2];
      size_t zz = (i%(transposed->dims[1]*transposed->dims[2]))%transposed->dims[2];

      printf("[%zu][%zu][%zu] - [%zu][%zu][%zu] =%lld - ", x, y, z,
      xx,yy,zz, i);

      int64_t new_index = data->dims[2]*z + y + data->dims[2]*data->dims[1]*x;
      //uint64_t new_index = z + transposed->dims[2]*y + transposed->dims[2]*transposed->dims[1]*x;
      printf("new index=%lld\n", new_index);
      transposed->float_data[i] = data->float_data[new_index];
    }*/



  return 0;
}

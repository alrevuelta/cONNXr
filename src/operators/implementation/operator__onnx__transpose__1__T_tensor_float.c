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

  if ((data->n_dims != 3) && (data->n_dims != 6)){
    printf("Dimensions not supported, only 3/6 not %zu\n", data->n_dims);
    exit(-1);
  }

  printf("first attribute %zu\n", ctx->onnx_node->n_attribute);
  printf("name value = %s\n", perm->name);
  printf("n_tensors %zu\n", perm->n_tensors);
  printf("type %u\n", perm->type);

  transposed->n_dims       = perm->n_ints;
  transposed->dims         = malloc(transposed->n_dims * sizeof(int64_t));
  transposed->has_raw_data = 0;
  transposed->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  transposed->n_float_data = data->n_float_data;
  transposed->float_data   = malloc(transposed->n_float_data * sizeof(float));

  // This should be the only case
  if (perm->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS){
    for (size_t i = 0; i < perm->n_ints; i++){
      transposed->dims[i] = data->dims[perm->ints[i]];
    }
  }

  /* TODO Quick solution. Only 3 and 6 are hardcoded. Rethink*/
  if (data->n_dims == 3){
    /*3d case*/
    size_t kk[3] = {data->dims[2]*data->dims[1], data->dims[2], 1};
    size_t kkk[3] = {data->dims[2]*data->dims[1], data->dims[2], 1};
    for (size_t i = 0; i < perm->n_ints; i++){
      kkk[i] = kk[perm->ints[i]];
    }
    for (size_t i = 0; i < transposed->dims[0]; i++){
      for (size_t j = 0; j < transposed->dims[1]; j++){
        for (size_t k = 0; k < transposed->dims[2]; k++){
          uint64_t out_index = k + transposed->dims[2]*j + transposed->dims[2]*transposed->dims[1]*i;
          uint64_t new_index = kkk[2]*k + kkk[1]*j + kkk[0]*i;
          transposed->float_data[out_index] = data->float_data[new_index];
        }
      }
    }
  }else if (data->n_dims == 6){
    /*6d case. Cool christmas tree */
    size_t kk[6] = {data->dims[5]*data->dims[4]*data->dims[3]*data->dims[2]*data->dims[1],
                    data->dims[5]*data->dims[4]*data->dims[3]*data->dims[2],
                    data->dims[5]*data->dims[4]*data->dims[3],
                    data->dims[5]*data->dims[4],
                    data->dims[5],
                    1};
    size_t kkk[6] = {data->dims[5]*data->dims[4]*data->dims[3]*data->dims[2]*data->dims[1],
                    data->dims[5]*data->dims[4]*data->dims[3]*data->dims[2],
                    data->dims[5]*data->dims[4]*data->dims[3],
                    data->dims[5]*data->dims[4],
                    data->dims[5],
                    1};
    for (size_t i = 0; i < perm->n_ints; i++){
      kkk[i] = kk[perm->ints[i]];
    }
    /*The alphabet has many letters, problem?*/
    for (size_t i = 0; i < transposed->dims[0]; i++){
      for (size_t j = 0; j < transposed->dims[1]; j++){
        for (size_t k = 0; k < transposed->dims[2]; k++){
          for (size_t l = 0; l < transposed->dims[3]; l++){
            for (size_t m = 0; m < transposed->dims[4]; m++){
              for (size_t n = 0; n < transposed->dims[5]; n++){
                uint64_t out_index = n +
                transposed->dims[5]*m +
                transposed->dims[5]*transposed->dims[4]*l +
                transposed->dims[5]*transposed->dims[4]*transposed->dims[3]*k +
                transposed->dims[5]*transposed->dims[4]*transposed->dims[3]*transposed->dims[2]*j +
                transposed->dims[5]*transposed->dims[4]*transposed->dims[3]*transposed->dims[2]*transposed->dims[1]*i;
                uint64_t new_index = kkk[5]*n + kkk[4]*m + kkk[3]*l + kkk[2]*k + kkk[1]*j + kkk[0]*i;
                transposed->float_data[out_index] = data->float_data[new_index];
              }
            }
          }
        }
      }
    }
  }


  debug_print_dims(transposed->n_dims, transposed->dims);
  return 0;
}

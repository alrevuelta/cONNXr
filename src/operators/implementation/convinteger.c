#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"
#include "operators.h"

int operator_convinteger(node_context *ctx)
{
  TRACE_LEVEL0("Calling operator_convinteger\n");

  /* TODO This is almost a copy paste from conv. Review!*/

  Onnx__TensorProto *X = searchInputByName(ctx, 0);
  Onnx__TensorProto *W = searchInputByName(ctx, 1);
  Onnx__TensorProto *x_zero_point = searchInputByName(ctx, 2);
  Onnx__TensorProto *w_zero_point = searchInputByName(ctx, 3);

  Onnx__TensorProto *y = searchOutputByName(ctx, 0);

  if (X->n_dims != 4){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // dilations is hardcoded ?

    //break if input is not int8 or uint8
    return -1;
  }

  debug_print_dims(X->n_dims, X->dims);

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  Onnx__AttributeProto *auto_pad = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "auto_pad");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "dilations");
  //Onnx__AttributeProto *group = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "pads");
  Onnx__AttributeProto *strides = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "strides");

  int64_t h_kernel, w_kernel, d_kernel, h_stride, w_stride;
  h_kernel = w_kernel = d_kernel = h_stride = w_stride = 1;
  if (kernel_shape != NULL) {
    h_kernel = kernel_shape->ints[0];
    w_kernel = kernel_shape->ints[1];
  }

  if (strides != NULL) {
    h_stride = strides->ints[0];
    w_stride = strides->ints[1];
  }

  // TODO Maybe use a smaller type
  int64_t h_pad, w_pad;
  h_pad = w_pad = 0;
  if (auto_pad != NULL){
    if (!strncmp((char*)auto_pad->s.data, "SAME_UPPER", 10)){
      if ("SAME_UPPER"){
        // This means, pad to match the output dimensions, and if not even
        // add the extra padding to the end
        // TODO Quick test, even padding is assumed
        // TODO Quick test, ignore stride, just 1
        h_pad = -(h_kernel - 1)/2; // store the negative value of the offset
        w_pad = -(w_kernel - 1)/2;
      }
    }
  }

  y->dims = malloc(X->n_dims * sizeof(int64_t));
  y->n_dims       = X->n_dims;
  // TODO Padding is not taken into account
  y->dims[0] = X->dims[0];

  /* Not sure about this. W might have different dimensions. This is if
  W has 4 dims (hardcoded for mnist model) */
  y->dims[1] = W->dims[0];
  //Y->dims[1] = X->dims[1];

  // TODO Formula is probably wrong, double check  // remove -
  y->dims[2] = (X->dims[2] - h_kernel + h_stride + -h_pad*2) / h_stride;
  y->dims[3] = (X->dims[3] - w_kernel + w_stride + -w_pad*2) / w_stride;

  y->has_raw_data = 0;
  y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT32;
  y->n_int32_data = y->dims[0]*y->dims[1]*y->dims[2]*y->dims[3];
  y->int32_data = malloc(y->n_int32_data * sizeof(int32_t));

  int b,i,j,k,m,n,d;
  for(b = 0; b < y->dims[0]; ++b){
    for(k = 0; k < y->dims[1]; ++k){
      for(i = 0; i < y->dims[2]; ++i){
        for(j = 0; j < y->dims[3]; ++j){
          // TODO replace all this calculations by macros?
          int out_index = j + y->dims[3]*(i + y->dims[2]*(k + X->dims[1]*b));
          int32_t value = 0;
          for(d = 0; d < W->dims[1]; ++d){
            for(n = 0; n < h_kernel; ++n){   // TODO use W->dims[2] instead?
              for(m = 0; m < w_kernel; ++m){ // TODO use W->dims[3] instead?
                int cur_h = i*h_stride + n + h_pad;
                int cur_w = j*w_stride + m + w_pad;

                /* This is hardcoded to make it work with mnist model, where
                the input is 1x1x28x28 */
                int index = cur_w + X->dims[3]*(cur_h + X->dims[2]*(d + 0*X->dims[1]));
                //TRACE_LEVEL0("%d,%d,%d index=%d\n", d, cur_h, cur_w, index);

                int valid = (cur_h >= 0 && cur_h < X->dims[2] &&
                             cur_w >= 0 && cur_w < X->dims[3]);

                int32_t valBeforeScaling = X->int32_data[index];

                /* TODO Ugly for performance, just testing*/
                if (ctx->onnx_node->n_input == 3){ // if x_zero_point. Wrong assumption, migh be w_zero_point, but quite rare
                  valBeforeScaling = valBeforeScaling + x_zero_point->int32_data[0];
                }
                if (ctx->onnx_node->n_input == 4){ // if w_zero_point
                  valBeforeScaling = valBeforeScaling * w_zero_point->int32_data[0];
                }

                int32_t val = (valid != 0) ? valBeforeScaling : 0;
                int index_kernel = k*W->dims[3]*W->dims[2]*W->dims[1] + d*W->dims[3]*W->dims[2] + n*h_kernel + m; // change h_kernel by W->dims[x]
                value += val * W->int32_data[index_kernel];
                //TRACE_LEVEL0("%fx%f+\n", val, W->float_data[index_kernel]);
              }
            }
          }
          y->int32_data[out_index] = value;

          /* TODO This is a huge crap to make it work with tinyYOLO
          It adds the bias, but this if will waste a lot of time. Make
          this nice!
          */
          /* Bias add is not implemented
          if (n_input == 3){
            y->int32_data[out_index] += x_zero_point->int32_data[k];
          }*/

        }
      }
    }
  }

  debug_print_dims(y->n_dims, y->dims);
  return 0;

}

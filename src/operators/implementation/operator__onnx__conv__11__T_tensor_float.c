#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

operator_status operator__onnx__conv__11__T_tensor_float(
    node_context *ctx
)
{
  TRACE_LEVEL0("Calling operator_conv\n");

  Onnx__TensorProto *X = searchInputByName(ctx, 0);
  Onnx__TensorProto *W = searchInputByName(ctx, 1);
  Onnx__TensorProto *B = searchInputByName(ctx, 2);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  if (X->n_dims != 4){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // dilations is hardcoded ?
    return -1;
  }

  debug_print_dims(X->n_dims, X->dims);

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  Onnx__AttributeProto *auto_pad = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "auto_pad");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "dilations");
  //Onnx__AttributeProto *group = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "pads");
  Onnx__AttributeProto *strides = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "strides");

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
      // This means, pad to match the output dimensions, and if not even
      // add the extra padding to the end
      // TODO Quick test, even padding is assumed
      // TODO Quick test, ignore stride, just 1
      h_pad = -(h_kernel - 1)/2; // store the negative value of the offset
      w_pad = -(w_kernel - 1)/2;
    }
  }

  Y->dims = malloc(X->n_dims * sizeof(int64_t));
  Y->n_dims       = X->n_dims;
  // TODO Padding is not taken into account
  Y->dims[0] = X->dims[0];

  /* Not sure about this. W might have different dimensions. This is if
  W has 4 dims (hardcoded for mnist model) */
  Y->dims[1] = W->dims[0];
  //Y->dims[1] = X->dims[1];

  // TODO Formula is probably wrong, double check  // remove -
  Y->dims[2] = (X->dims[2] - h_kernel + h_stride + -h_pad*2) / h_stride;
  Y->dims[3] = (X->dims[3] - w_kernel + w_stride + -w_pad*2) / w_stride;

  Y->has_raw_data = 0;

  // TODO
  Y->data_type = X->data_type;

  Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  Y->n_float_data = Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3];

  //TODO This is wrong. n_dims can be like 2 and this will fail
  TRACE_LEVEL0("n_flot_data = %zu\n", X->n_dims);

  Y->float_data = malloc(Y->n_float_data * sizeof(float));

  int b,i,j,k,m,n,d;
  for(b = 0; b < Y->dims[0]; ++b){
    for(k = 0; k < Y->dims[1]; ++k){
      for(i = 0; i < Y->dims[2]; ++i){
        for(j = 0; j < Y->dims[3]; ++j){
          // TODO replace all this calculations by macros?
          int out_index = j + Y->dims[3]*(i + Y->dims[2]*(k + X->dims[1]*b));
          float value = 0;
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
                float val = (valid != 0) ? X->float_data[index] : 0;
                int index_kernel = k*W->dims[3]*W->dims[2]*W->dims[1] + d*W->dims[3]*W->dims[2] + n*h_kernel + m; // change h_kernel by W->dims[x]
                value += val * W->float_data[index_kernel];
                //TRACE_LEVEL0("%fx%f+\n", val, W->float_data[index_kernel]);
              }
            }
          }
          Y->float_data[out_index] = value;

          /* TODO This is a huge crap to make it work with tinyYOLO
          It adds the bias, but this if will waste a lot of time. Make
          this nice!
          */
          if (ctx->onnx_node->n_input == 3){
            Y->float_data[out_index] += B->float_data[k];
          }
        }
      }
    }
  }

  debug_print_dims(Y->n_dims, Y->dims);
  return 0;
}

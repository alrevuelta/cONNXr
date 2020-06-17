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

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // dilations is hardcoded ?
    fprintf(stderr, "Not implemented\n");
    exit(-1);
  }

  debug_print_dims(X->n_dims, X->dims);

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  Onnx__AttributeProto *auto_pad     = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "auto_pad");
  Onnx__AttributeProto *dilations    = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "dilations");
  Onnx__AttributeProto *group        = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "kernel_shape");
  Onnx__AttributeProto *pads         = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "pads");
  Onnx__AttributeProto *strides      = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "strides");

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
      h_pad = (h_kernel - 1)/2;
      w_pad = (w_kernel - 1)/2;
    }
  }

  if (pads != NULL){
    /* Will work oly with the cases where the padding is symetric */
    h_pad = pads->ints[0];
    w_pad = pads->ints[2];
  }

  Y->n_dims = W->n_dims;
  Y->dims   = malloc(Y->n_dims * sizeof(int64_t));

  if (X->n_dims == 4){
    Y->dims[0] = X->dims[0];
    Y->dims[1] = W->dims[0];
    Y->dims[2] = (X->dims[2] - h_kernel + h_stride + h_pad*2) / h_stride;
    Y->dims[3] = (X->dims[3] - w_kernel + w_stride + w_pad*2) / w_stride;
  }else if (X->n_dims == 2){
    Y->dims[0] = W->dims[0];
    Y->dims[1] = W->dims[1];
    Y->dims[2] = (X->dims[0] - h_kernel + h_stride + h_pad*2) / h_stride;
    Y->dims[3] = (X->dims[1] - w_kernel + w_stride + w_pad*2) / w_stride;
  }else if (X->n_dims == 3){
    // TODO
  }

  Y->has_raw_data = 0;
  Y->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  Y->n_float_data = Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3];
  Y->float_data   = malloc(Y->n_float_data * sizeof(float));

  int b,i,j,k,m,n,d;
  for(b = 0; b < Y->dims[0]; ++b){
    for(k = 0; k < Y->dims[1]; ++k){
      for(i = 0; i < Y->dims[2]; ++i){
        for(j = 0; j < Y->dims[3]; ++j){
          // TODO replace all this calculations by macros?
          uint64_t out_index = j + Y->dims[3]*(i + Y->dims[2]*(k + Y->dims[1]*b));
          float value = 0;

          if (X->n_dims == 4){
            for(d = 0; d < W->dims[1]; ++d){
              for(n = 0; n < h_kernel; ++n){   // TODO use W->dims[2] instead?
                for(m = 0; m < w_kernel; ++m){ // TODO use W->dims[3] instead?
                  int cur_h = i*h_stride + n - h_pad;
                  int cur_w = j*w_stride + m - w_pad;

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
          }else if (X->n_dims == 2){
            for(n = 0; n < h_kernel; ++n){
              for(m = 0; m < w_kernel; ++m){
                int cur_h = i*h_stride + n - h_pad;
                int cur_w = j*w_stride + m - w_pad;
                //printf("%d, %d\n", cur_h, cur_w);
                int index = cur_w + X->dims[1]*cur_h;
                //printf("index=%d\n", index);
                int valid = (cur_h >= 0 && cur_h < X->dims[0] &&
                             cur_w >= 0 && cur_w < X->dims[1]);
                float val = (valid != 0) ? X->float_data[index] : 0;
                int index_kernel = k*W->dims[3]*W->dims[2]*W->dims[1] + n*h_kernel + m;
                value += val * W->float_data[index_kernel];
                //TRACE_LEVEL0("%fx%f+\n", val, input[1]->float_data[index_kernel]);
              }
            }
          }else{

          }


          Y->float_data[out_index] = value;
          //printf("%lld\n", out_index);
          printf("[%lld]=%f\n", out_index, value);

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

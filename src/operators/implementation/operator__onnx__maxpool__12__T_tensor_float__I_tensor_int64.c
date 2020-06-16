#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

operator_status operator__onnx__maxpool__12__T_tensor_float__I_tensor_int64(
    node_context *ctx
)
{
  TRACE_LEVEL0("Calling operator_maxpool\n");

  Onnx__TensorProto *X = searchInputByName(ctx, 0);

  Onnx__TensorProto *Y = searchOutputByName(ctx, 0);
  //Onnx__TensorProto *Indices = searchOutputByName(ctx, 1);

  debug_print_dims(X->n_dims, X->dims);
  //debug_print_attributes(n_attribute, attribute);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    // TODO ingore dilated parameter for initial tests
    // TODO indices are not implemented for the initial prototype
    // TODO this is hardcoded hardcoded for 2d (dims = 4)
    return 1;
  }

  // number of dimensions do not change
  Y->dims   = malloc(X->n_dims * sizeof(int64_t));
  Y->n_dims = X->n_dims;

  // Only kernel_shape is mandatory
  Onnx__AttributeProto *auto_pad = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "auto_pad");
  //Onnx__AttributeProto *ceil_mode = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "ceil_mode");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "dilations");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "kernel_shape");
  Onnx__AttributeProto *pads = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "pads");
  //Onnx__AttributeProto *storage_order = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "storage_order");
  Onnx__AttributeProto *strides = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "strides");

  int64_t h_kernel, w_kernel, h_stride, w_stride;
  h_kernel = w_kernel = h_stride = w_stride = 1;
  if (kernel_shape != NULL) {
    h_kernel = kernel_shape->ints[0];
    w_kernel = kernel_shape->ints[1];
  }

  if (strides != NULL) {
    h_stride = strides->ints[0];
    w_stride = strides->ints[1];
  }

  // TODO Maybe use a smaller type
  /* left and right pads */
  int h_pad, w_pad;
  h_pad = w_pad = 0;

  int h_pad_aux = 0;
  int w_pad_aux = 0;
  if (auto_pad != NULL){
    h_pad_aux = (h_kernel - 1);
    w_pad_aux = (w_kernel - 1);
    h_pad = (h_kernel - 1)/2;
    w_pad = (w_kernel - 1)/2;
    if (!strncmp((char*)auto_pad->s.data, "SAME_UPPER", 10)){
      // remove
    } else if (!strncmp((char*)auto_pad->s.data, "SAME_LOWER", 10)){
      /* TODO quick n dirty*/
      if ((h_kernel - 1)%2 != 0){
        h_pad++;
      }
      if ((w_kernel - 1)%2 != 0){
        w_pad++;
      }
    }
  }

  if (pads != NULL){
    /* TODO */
    /* Hardcoded for pads = [x, x, x, x] dim = 4*/
    h_pad_aux = pads->ints[0] + pads->ints[2];
    w_pad_aux = pads->ints[1] + pads->ints[3];
    h_pad = h_pad_aux/2;
    w_pad = w_pad_aux/2;

    /*
    for (int i = 0; i < pads->n_ints; i++){
      TRACE_LEVEL0("\n\n pad=%d\n", pads->ints[i]);
    }*/
  }

  Y->dims[0] = X->dims[0];
  Y->dims[1] = X->dims[1];
  Y->dims[2] = (int64_t)floorf((float)(X->dims[2] + h_pad_aux - ((h_kernel - 1) + 1)) / (float)h_stride + 1);
  Y->dims[3] = (int64_t)floorf((float)(X->dims[3] + w_pad_aux - ((w_kernel - 1) + 1)) / (float)w_stride + 1);
  Y->has_raw_data = 0;
  Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
  Y->float_data = malloc(Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3] * sizeof(float));
  Y->n_float_data = Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3];

  int b,i,j,k,m,n;
  for(b = 0; b < Y->dims[0]; ++b){
    for(k = 0; k < Y->dims[1]; ++k){
      for(i = 0; i < Y->dims[2]; ++i){
        for(j = 0; j < Y->dims[3]; ++j){
          int out_index = j + Y->dims[3]*(i + Y->dims[2]*(k + X->dims[1]*b));
          float max = -999999; // TODO
          for(n = 0; n < h_kernel; ++n){
            for(m = 0; m < w_kernel; ++m){
              int cur_h = i*h_stride + n -h_pad;
              int cur_w = j*w_stride + m -w_pad;
              int index = cur_w + X->dims[3]*(cur_h + X->dims[2]*(k + b*X->dims[1]));
              int valid = (cur_h >= 0 && cur_h < (X->dims[2]) &&
                           cur_w >= 0 && cur_w < (X->dims[3]));
              float val = (valid != 0) ? X->float_data[index] : -999999; //TODO
              max = (val > max ? val : max);
              }
            }
            Y->float_data[out_index] = max;
          }
        }
      }
    }

  debug_print_dims(Y->n_dims, Y->dims);
  return 0;

}

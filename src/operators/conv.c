#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "../embeddedml_utils.h"
#include "conv.h"

// Template example
/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  xx xx
 *  \param[in]  xx xx
 *  \param[out] xx xx
 *  \return     xx
 */
 void operator_conv(Onnx__TensorProto *X,
                    Onnx__TensorProto *W,
                    Onnx__TensorProto *B,
                    Onnx__TensorProto *Y,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute)
{
  DEBUG_PRINT("Calling operator_conv");
  debug_print_dims(X->n_dims, X->dims);
  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  // TODO dilations is harcoded [1 1]
  // TODO strides is hardcoded [1 1]
  // TODO group is hardcoded 1
  // Hardcoded for 4d (2d)

  Onnx__AttributeProto *auto_pad = searchAttributeNyName(n_attribute, attribute, "auto_pad");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(n_attribute, attribute, "dilations");
  //Onnx__AttributeProto *group = searchAttributeNyName(n_attribute, attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(n_attribute, attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(n_attribute, attribute, "pads");
  Onnx__AttributeProto *strides = searchAttributeNyName(n_attribute, attribute, "strides");

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
  int64_t h_pad, w_pad;
  h_pad = w_pad = 0;
  if (!strcmp((const char*)auto_pad->s.data, "SAME_UPPER")){
    if ("SAME_UPPER"){
      // This means, pad to match the output dimensions, and if not even
      // add the extra padding to the end
      // TODO Quick test, even padding is assumed
      // TODO Quick test, ignore stride, just 1
      h_pad = -(h_kernel - 1); // store the negative value of the offset
      w_pad = -(w_kernel - 1);
    }

  }

  Y->dims = malloc(X->n_dims * sizeof(int64_t));
  Y->n_dims       = X->n_dims;

  // TODO Padding is not taken into account
  Y->dims[0] = X->dims[0];
  Y->dims[1] = X->dims[1];

  // TODO Formula is probably wrong, double check  // remove -
  Y->dims[2] = (X->dims[2] - h_kernel + h_stride + -h_pad) / h_stride;
  Y->dims[3] = (X->dims[3] - w_kernel + w_stride + -w_pad) / w_stride;

  // TODO check this? no mem is allocated?
  Y->name         = "name_is_set_afterwards\0";
  Y->has_raw_data = 0;

  switch(X->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      Y->float_data = malloc(Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3] * sizeof(float));
      Y->n_float_data = Y->dims[0]*Y->dims[1]*Y->dims[2]*Y->dims[3];

      int b,i,j,k,m,n;
      for(b = 0; b < Y->dims[0]; ++b){
        for(k = 0; k < Y->dims[1]; ++k){
          for(i = 0; i < Y->dims[2]; ++i){
            for(j = 0; j < Y->dims[3]; ++j){
              int out_index = j + Y->dims[3]*(i + Y->dims[2]*(k + X->dims[1]*b));
              float value = 0;
              for(n = 0; n < h_kernel; ++n){
                for(m = 0; m < w_kernel; ++m){
                  // TODO Not sure about this. The idea is to have here a negative
                  // number if we are on a padded "pixel". Left side is ok but
                  // not sure about the right side
                  int cur_h = h_pad + i*h_stride + n;
                  int cur_w = w_pad + j*w_stride + m;
                  int index = cur_w + X->dims[3]*(cur_h + X->dims[2]*(k + b*X->dims[1]));
                  int valid = (cur_h >= 0 && cur_h < X->dims[2] &&
                               cur_w >= 0 && cur_w < X->dims[3]);
                  // Padded with 0, is this right?
                  float val = (valid != 0) ? X->float_data[index] : 0;
                  value += val * W->float_data[n*h_kernel + m];
                  //printf("mult %f * %f\n", val, W->float_data[n*h_kernel + m]);
                  }
                }
                Y->float_data[out_index] = value;
              }
            }
          }
        }
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      break;
    default:
      break;
  }

  debug_print_dims(Y->n_dims, Y->dims);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "../embeddedml_utils.h"
#include "maxpool.h"

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
 void operator_maxpool(Onnx__TensorProto *X,
                       Onnx__TensorProto *Y,
                       Onnx__TensorProto *Indices,
                       size_t n_attribute,
                       Onnx__AttributeProto **attribute)
{
  DEBUG_PRINT("Calling operator_maxpool");
  //debug_print_attributes(n_attribute, attribute);

  // TODO ingore dilated parameter for initial tests
  // TODO indices are not implemented for the initial prototype
  // TODO this is hardcoded af. only for 4d arrays, where maxpool
  // is applied along 2dimensions.
  // TODO pads are not implemented

  // number of dimensions do not change
  Y->dims = malloc(X->n_dims * sizeof(int64_t));
  Y->n_dims       = X->n_dims;

  // Only kernel_shape is mandatory
  //Onnx__AttributeProto *auto_pad = searchAttributeNyName(n_attribute, attribute, "auto_pad");
  //Onnx__AttributeProto *ceil_mode = searchAttributeNyName(n_attribute, attribute, "ceil_mode");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(n_attribute, attribute, "dilations");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(n_attribute, attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(n_attribute, attribute, "pads");
  //Onnx__AttributeProto *storage_order = searchAttributeNyName(n_attribute, attribute, "storage_order");
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

  Y->dims[0] = X->dims[0];
  Y->dims[1] = X->dims[1];
  Y->dims[2] = (X->dims[2] - h_kernel + h_stride) / h_stride;
  Y->dims[3] = (X->dims[3] - w_kernel + w_stride) / w_stride;

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
              float max = -99999; // TODO
              for(n = 0; n < h_kernel; ++n){
                for(m = 0; m < w_kernel; ++m){
                  int cur_h = i*h_stride + n;
                  int cur_w = j*w_stride + m;
                  int index = cur_w + X->dims[3]*(cur_h + X->dims[2]*(k + b*X->dims[1]));
                  /*int valid = (cur_h >= 0 && cur_h < l.h &&
                               cur_w >= 0 && cur_w < l.w);
                  float val = (valid != 0) ? net.input[index] : -FLT_MAX;*/
                  max = (X->float_data[index] > max ? X->float_data[index] : max);
                  //max_i = (val > max) ? index : max_i;
                  //max   = (val > max) ? val   : max;
                  }
                }
                Y->float_data[out_index] = max;
              }
            }
          }
        }
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    default:
      break;
  }

}

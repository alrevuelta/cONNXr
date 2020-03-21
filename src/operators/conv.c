#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../trace.h"
#include "../utils.h"
#include "operators.h"

/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *  Limitations: There might be some limitations with respect to the official onnx
 *  operator. Write here possible limitations, i.e. if the function doesnt work
 *  with all types, or if it works with a specific number of dimensions only
 *
 *  \param[in]      n_input     Number of inputs of the operator
 *  \param[in]      input       Array of pointers to the inputs of the operator
 *  \param[in]      n_attribute Number of attributes of the operator
 *  \param[in]      attribute   Array of pointers to the attributes of the operator
 *  \param[in]      n_output    Numper of outputs of the operator
 *  \param[in/out]  output      Array of pointer to the outputs of the operators
 *  \return         error       Different than 0 if an error was produced
 */
 int operator_conv(size_t n_input,
                   Onnx__TensorProto **input,
                   size_t n_attribute,
                   Onnx__AttributeProto **attribute,
                   size_t n_output,
                   Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_conv\n");

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
  }

  debug_print_dims(input[0]->n_dims, input[0]->dims);
  debug_print_dims(input[1]->n_dims, input[1]->dims);

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  Onnx__AttributeProto *auto_pad     = searchAttributeNyName(n_attribute, attribute, "auto_pad");
  Onnx__AttributeProto *dilations    = searchAttributeNyName(n_attribute, attribute, "dilations");
  Onnx__AttributeProto *group        = searchAttributeNyName(n_attribute, attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(n_attribute, attribute, "kernel_shape");
  Onnx__AttributeProto *pads         = searchAttributeNyName(n_attribute, attribute, "pads");
  Onnx__AttributeProto *strides      = searchAttributeNyName(n_attribute, attribute, "strides");

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
      h_pad = (h_kernel - 1)/2; // store the negative value of the offset
      w_pad = (w_kernel - 1)/2;
    }
  }

  if (pads != NULL){
    /* Will work oly with the cases where the padding is symetric */
    h_pad = pads->ints[0];
    w_pad = pads->ints[2];
  }

  output[0]->n_dims = input[1]->n_dims;
  output[0]->dims   = malloc(output[0]->n_dims * sizeof(int64_t));
  output[0]->has_raw_data = 0;

  if (input[0]->n_dims == 4){
    output[0]->dims[0] = input[0]->dims[0];
    output[0]->dims[1] = input[1]->dims[0];
    output[0]->dims[2] = (input[0]->dims[2] - h_kernel + h_stride - h_pad*2) / h_stride;
    output[0]->dims[3] = (input[0]->dims[3] - w_kernel + w_stride - w_pad*2) / w_stride;
  }else if (input[0]->n_dims == 2){
    output[0]->dims[0] = input[1]->dims[0];
    output[0]->dims[1] = input[1]->dims[1];
    output[0]->dims[2] = (input[0]->dims[0] - h_kernel + h_stride - h_pad*2) / h_stride;
    output[0]->dims[3] = (input[0]->dims[1] - w_kernel + w_stride - w_pad*2) / w_stride;
  }else if (input[0]->n_dims == 3){
    // TODO
    return -1;
  }

  switch(input[0]->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->n_float_data = output[0]->dims[0]*output[0]->dims[1]*output[0]->dims[2]*output[0]->dims[3];
      output[0]->float_data = malloc(output[0]->n_float_data * sizeof(float));

      int b,i,j,k,m,n,d;
      for(b = 0; b < output[0]->dims[0]; ++b){
        for(k = 0; k < output[0]->dims[1]; ++k){
          for(i = 0; i < output[0]->dims[2]; ++i){
            for(j = 0; j < output[0]->dims[3]; ++j){
              // TODO replace all this calculations by macros?
              uint64_t out_index = j + output[0]->dims[3]*(i + output[0]->dims[2]*(k + output[0]->dims[1]*b));
              float value = 0;

              if (input[0]->n_dims == 4){
                for(d = 0; d < input[1]->dims[1]; ++d){ //?
                  for(n = 0; n < h_kernel; ++n){
                    for(m = 0; m < w_kernel; ++m){
                      int cur_h = i*h_stride + n - h_pad;
                      int cur_w = j*w_stride + m - w_pad;

                      /* This is hardcoded to make it work with mnist model, where
                      the input is 1x1x28x28 */
                      int index = cur_w + input[0]->dims[3]*(cur_h + input[0]->dims[2]*(d + 0*input[0]->dims[1]));
                      int valid = (cur_h >= 0 && cur_h < input[0]->dims[2] &&
                                   cur_w >= 0 && cur_w < input[0]->dims[3]);
                      float val = (valid != 0) ? input[0]->float_data[index] : 0;
                      int index_kernel = k*input[1]->dims[3]*input[1]->dims[2]*input[1]->dims[1] + d*input[1]->dims[3]*input[1]->dims[2] + n*h_kernel + m; // change h_kernel by W->dims[x]
                      value += val * input[1]->float_data[index_kernel];
                    }
                  }
                }

              }else if (input[0]->n_dims == 2){
                //printf("enter %d\n", out_index);
                for(n = 0; n < h_kernel; ++n){
                  for(m = 0; m < w_kernel; ++m){
                    int cur_h = i*h_stride + n - h_pad;
                    int cur_w = j*w_stride + m - w_pad;
                    //printf("%d, %d\n", cur_h, cur_w);
                    int index = cur_w + input[0]->dims[1]*cur_h;
                    //printf("index=%d\n", index);
                    int valid = (cur_h >= 0 && cur_h < input[0]->dims[0] &&
                                 cur_w >= 0 && cur_w < input[0]->dims[1]);
                    float val = (valid != 0) ? input[0]->float_data[index] : 0;
                    int index_kernel = k*input[1]->dims[3]*input[1]->dims[2]*input[1]->dims[1] + n*h_kernel + m;
                    value += val * input[1]->float_data[index_kernel];
                    //TRACE_LEVEL0("%fx%f+\n", val, input[1]->float_data[index_kernel]);
                  }
                }

              }else{
                // TODO?
                return -1;
              }

              output[0]->float_data[out_index] = value;

              /* TODO This is a huge crap to make it work with tinyYOLO
              It adds the bias, but this if will waste a lot of time. Make
              this nice!
              */
              if (n_input == 3){
                output[0]->float_data[out_index] += input[2]->float_data[k];
              }
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

  debug_print_dims(output[0]->n_dims, output[0]->dims);
  return 0;
}

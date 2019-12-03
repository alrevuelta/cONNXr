#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "../embeddedml_utils.h"
#include "conv.h"

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
 int operator_conv(const size_t n_input,
                   const Onnx__TensorProto **input,
                   const size_t n_attribute,
                   const Onnx__AttributeProto **attribute,
                   const size_t n_output,
                   Onnx__TensorProto **output)
{
  DEBUG_PRINT("Calling operator_conv");
  debug_print_dims(input[0]->n_dims, input[0]->dims);

  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // dilations is hardcoded ?
    return -1;
  }

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  Onnx__AttributeProto *auto_pad = searchAttributeNyName(n_attribute, attribute, "auto_pad");
  //Onnx__AttributeProto *dilations = searchAttributeNyName(n_attribute, attribute, "dilations");
  //Onnx__AttributeProto *group = searchAttributeNyName(n_attribute, attribute, "group");
  Onnx__AttributeProto *kernel_shape = searchAttributeNyName(n_attribute, attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(n_attribute, attribute, "pads");
  Onnx__AttributeProto *strides = searchAttributeNyName(n_attribute, attribute, "strides");

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
    if (!strcmp((const char*)auto_pad->s.data, "SAME_UPPER")){
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

  output[0]->dims = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims       = input[0]->n_dims;
  // TODO Padding is not taken into account
  output[0]->dims[0] = input[0]->dims[0];

  /* Not sure about this. W might have different dimensions. This is if
  W has 4 dims (hardcoded for mnist model) */
  output[0]->dims[1] = input[1]->dims[0];
  //Y->dims[1] = input[0]->dims[1];

  // TODO Formula is probably wrong, double check  // remove -
  output[0]->dims[2] = (input[0]->dims[2] - h_kernel + h_stride + -h_pad*2) / h_stride;
  output[0]->dims[3] = (input[0]->dims[3] - w_kernel + w_stride + -w_pad*2) / w_stride;

  // TODO check this? no mem is allocated?
  output[0]->name         = "name_is_set_afterwards\0"; // todo this is wrong.
  output[0]->has_raw_data = 0;

  switch(input[0]->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->float_data = malloc(output[0]->dims[0]*output[0]->dims[1]*output[0]->dims[2]*output[0]->dims[3] * sizeof(float));
      // todo make this nice :P
      output[0]->n_float_data = output[0]->dims[0]*output[0]->dims[1]*output[0]->dims[2]*output[0]->dims[3];

      int b,i,j,k,m,n,d;
      for(b = 0; b < output[0]->dims[0]; ++b){
        for(k = 0; k < output[0]->dims[1]; ++k){
          for(i = 0; i < output[0]->dims[2]; ++i){
            for(j = 0; j < output[0]->dims[3]; ++j){
              // TODO replace all this calculations by macros?
              int out_index = j + output[0]->dims[3]*(i + output[0]->dims[2]*(k + input[0]->dims[1]*b));
              float value = 0;
              for(d = 0; d < input[1]->dims[1]; ++d){
                for(n = 0; n < h_kernel; ++n){   // TODO use W->dims[2] instead?
                  for(m = 0; m < w_kernel; ++m){ // TODO use W->dims[3] instead?
                    int cur_h = i*h_stride + n + h_pad;
                    int cur_w = j*w_stride + m + w_pad;

                    /* This is hardcoded to make it work with mnist model, where
                    the input is 1x1x28x28 */
                    int index = cur_w + input[0]->dims[3]*(cur_h + input[0]->dims[2]*(d + 0*input[0]->dims[1]));
                    //printf("%d,%d,%d index=%d\n", d, cur_h, cur_w, index);

                    int valid = (cur_h >= 0 && cur_h < input[0]->dims[2] &&
                                 cur_w >= 0 && cur_w < input[0]->dims[3]);
                    float val = (valid != 0) ? input[0]->float_data[index] : 0;
                    int index_kernel = k*input[1]->dims[3]*input[1]->dims[2]*input[1]->dims[1] + d*input[1]->dims[3]*input[1]->dims[2] + n*h_kernel + m; // change h_kernel by W->dims[x]
                    value += val * input[1]->float_data[index_kernel];
                    //printf("%fx%f+\n", val, input[1]->float_data[index_kernel]);
                  }
                }
              }
              output[0]->float_data[out_index] = value;
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

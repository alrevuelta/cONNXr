#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "../embeddedml_utils.h"
#include "maxpool.h"

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
int operator_maxpool(const size_t n_input,
                     const Onnx__TensorProto **input,
                     const size_t n_attribute,
                     const Onnx__AttributeProto **attribute,
                     const size_t n_output,
                     Onnx__TensorProto **output)
{
  DEBUG_PRINT("Calling operator_maxpool");
  debug_print_dims(input[0]->n_dims, input[0]->dims);
  //debug_print_attributes(n_attribute, attribute);

  /* Use conv operator as reference to improve this one */
  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    return -1;
  }

  // TODO ingore dilated parameter for initial tests
  // TODO indices are not implemented for the initial prototype
  // TODO this is hardcoded af. only for 4d arrays, where maxpool
  // is applied along 2dimensions.
  // TODO pads are not implemented

  // number of dimensions do not change
  output[0]->dims   = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims = input[0]->n_dims;

  // Only kernel_shape is mandatory
  Onnx__AttributeProto *auto_pad = searchAttributeNyName(n_attribute, attribute, "auto_pad");
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

  // TODO Maybe use a smaller type
  /* left and right pads */
  int64_t h_pad, w_pad;
  h_pad = w_pad = 0;

  // rmeove
  int h_pad_aux = 0;
  int w_pad_aux = 0;
  if (auto_pad != NULL){
    if (!strcmp((const char*)auto_pad->s.data, "SAME_UPPER")){
      h_pad = (h_kernel - 1)/2;
      w_pad = (w_kernel - 1)/2;
      printf("h_kernel = %lld\n", h_kernel);
      printf("w_kernel = %lld\n", w_kernel);
      h_pad_aux = (h_kernel - 1);
      w_pad_aux = (w_kernel - 1);
    } else if (!strcmp((const char*)auto_pad->s.data, "SAME_LOWER")){

      /*h_pad = (h_kernel - 1)/2;
      w_pad = (w_kernel - 1)/2;
      printf("h_kernel = %lld\n", h_kernel);
      printf("w_kernel = %lld\n", w_kernel);
      h_pad_aux = (h_kernel - 1);
      w_pad_aux = (w_kernel - 1);*/
    }
  }

  printf("h_pad=%lld, w_pad=%lld\n", h_pad, w_pad);
  printf("h_padaux=%lld, w_padaux=%lld\n", h_pad_aux, w_pad_aux);

  output[0]->dims[0] = input[0]->dims[0];
  output[0]->dims[1] = input[0]->dims[1];

/* TODO Use this instead. floor or ceil
  floorf((float)(input[0]->dims[2] + h_pad_aux - ((h_kernel - 1) + 1)) / (float)h_stride + 1);
  floorf((float)(input[0]->dims[3] + w_pad_aux - ((w_kernel - 1) + 1)) / (float)w_stride + 1);
  */

  output[0]->dims[2] = (input[0]->dims[2] - h_kernel + h_stride + h_pad_aux) / h_stride;
  output[0]->dims[3] = (input[0]->dims[3] - w_kernel + w_stride + w_pad_aux) / w_stride;


  // TODO check this? no mem is allocated?
  output[0]->name         = "name_is_set_afterwards\0"; // dont do this
  output[0]->has_raw_data = 0;

  switch(input[0]->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      output[0]->float_data = malloc(output[0]->dims[0]*output[0]->dims[1]*output[0]->dims[2]*output[0]->dims[3] * sizeof(float));
      output[0]->n_float_data = output[0]->dims[0]*output[0]->dims[1]*output[0]->dims[2]*output[0]->dims[3];

      int b,i,j,k,m,n;
      for(b = 0; b < output[0]->dims[0]; ++b){
        for(k = 0; k < output[0]->dims[1]; ++k){
          for(i = 0; i < output[0]->dims[2]; ++i){
            for(j = 0; j < output[0]->dims[3]; ++j){
              int out_index = j + output[0]->dims[3]*(i + output[0]->dims[2]*(k + input[0]->dims[1]*b));
              float max = -999999; // TODO
              for(n = 0; n < h_kernel; ++n){
                for(m = 0; m < w_kernel; ++m){
                  int cur_h = i*h_stride + n -h_pad;
                  int cur_w = j*w_stride + m -w_pad;
                  int index = cur_w + input[0]->dims[3]*(cur_h + input[0]->dims[2]*(k + b*input[0]->dims[1]));
                  int valid = (cur_h >= 0 && cur_h < (input[0]->dims[2]) &&
                               cur_w >= 0 && cur_w < (input[0]->dims[3]));
                  float val = (valid != 0) ? input[0]->float_data[index] : -999999; //TODO
                  max = (val > max ? val : max);
                  }
                }
                output[0]->float_data[out_index] = max;
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

  debug_print_dims(output[0]->n_dims, output[0]->dims);
  return 0;

}

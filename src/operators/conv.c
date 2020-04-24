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
 int operator_conv(struct operator__context *context)
{
  TRACE_LEVEL0("Calling operator_conv\n");

  struct operator__onnx__conv__context *sc = (void *) context;

  if (sc->in->X->n_dims != 4){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    //a->data_type == b->data_type
    //a->n_dims == b->n_dims
    //a->dims[i] == b->dims[i]
    // dilations is hardcoded ?
    return -1;
  }



  debug_print_dims(sc->in->X->n_dims, sc->in->X->dims);

  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  //Onnx__AttributeProto *sc->attr->auto_pad = searchAttributeNyName(n_attribute, attribute, "auto_pad");
  //Onnx__AttributeProto *sc->out->B = searchAttributeNyName(n_attribute, attribute, "sc->out->B");
  //Onnx__AttributeProto *group = searchAttributeNyName(n_attribute, attribute, "group");
  //Onnx__AttributeProto *kernel_shape = searchAttributeNyName(n_attribute, attribute, "kernel_shape");
  //Onnx__AttributeProto *pads = searchAttributeNyName(n_attribute, attribute, "pads");
  //Onnx__AttributeProto *strides = searchAttributeNyName(n_attribute, attribute, "strides");

  int64_t h_kernel, w_kernel, d_kernel, h_stride, w_stride;
  h_kernel = w_kernel = d_kernel = h_stride = w_stride = 1;
  if (sc->attr->kernel_shape != NULL) {
    h_kernel = sc->attr->kernel_shape->ints[0];
    w_kernel = sc->attr->kernel_shape->ints[1];
  }

  if (sc->attr->strides != NULL) {
    h_stride = sc->attr->strides->ints[0];
    w_stride = sc->attr->strides->ints[1];
  }

  // TODO Maybe use a smaller type
  int64_t h_pad, w_pad;
  h_pad = w_pad = 0;
  if (sc->attr->auto_pad != NULL){
    if (!strncmp((char*)sc->attr->auto_pad->s.data, "SAME_UPPER", 10)){
      // This means, pad to match the output dimensions, and if not even
      // add the extra padding to the end
      // TODO Quick test, even padding is assumed
      // TODO Quick test, ignore stride, just 1
      h_pad = -(h_kernel - 1)/2; // store the negative value of the offset
      w_pad = -(w_kernel - 1)/2;
    }
  }

  sc->out->Y->dims = malloc(sc->in->X->n_dims * sizeof(int64_t));
  sc->out->Y->n_dims       = sc->in->X->n_dims;
  // TODO Padding is not taken into account
  sc->out->Y->dims[0] = sc->in->X->dims[0];

  /* Not sure about this. W might have different dimensions. This is if
  W has 4 dims (hardcoded for mnist model) */
  sc->out->Y->dims[1] = sc->in->W->dims[0];
  //Y->dims[1] = sc->in->X->dims[1];

  // TODO Formula is probably wrong, double check  // remove -
  sc->out->Y->dims[2] = (sc->in->X->dims[2] - h_kernel + h_stride + -h_pad*2) / h_stride;
  sc->out->Y->dims[3] = (sc->in->X->dims[3] - w_kernel + w_stride + -w_pad*2) / w_stride;

  sc->out->Y->has_raw_data = 0;

  switch(sc->in->X->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      sc->out->Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      sc->out->Y->n_float_data = sc->out->Y->dims[0]*sc->out->Y->dims[1]*sc->out->Y->dims[2]*sc->out->Y->dims[3];

      //TODO This is wrong. n_dims can be like 2 and this will fail
      TRACE_LEVEL0("n_flot_data = %zu\n", sc->in->X->n_dims);

      sc->out->Y->float_data = malloc(sc->out->Y->n_float_data * sizeof(float));

      int b,i,j,k,m,n,d;
      for(b = 0; b < sc->out->Y->dims[0]; ++b){
        for(k = 0; k < sc->out->Y->dims[1]; ++k){
          for(i = 0; i < sc->out->Y->dims[2]; ++i){
            for(j = 0; j < sc->out->Y->dims[3]; ++j){
              // TODO replace all this calculations by macros?
              int out_index = j + sc->out->Y->dims[3]*(i + sc->out->Y->dims[2]*(k + sc->in->X->dims[1]*b));
              float value = 0;
              for(d = 0; d < sc->in->W->dims[1]; ++d){
                for(n = 0; n < h_kernel; ++n){   // TODO use W->dims[2] instead?
                  for(m = 0; m < w_kernel; ++m){ // TODO use W->dims[3] instead?
                    int cur_h = i*h_stride + n + h_pad;
                    int cur_w = j*w_stride + m + w_pad;

                    /* This is hardcoded to make it work with mnist model, where
                    the input is 1x1x28x28 */
                    int index = cur_w + sc->in->X->dims[3]*(cur_h + sc->in->X->dims[2]*(d + 0*sc->in->X->dims[1]));
                    //TRACE_LEVEL0("%d,%d,%d index=%d\n", d, cur_h, cur_w, index);

                    int valid = (cur_h >= 0 && cur_h < sc->in->X->dims[2] &&
                                 cur_w >= 0 && cur_w < sc->in->X->dims[3]);
                    float val = (valid != 0) ? sc->in->X->float_data[index] : 0;
                    int index_kernel = k*sc->in->W->dims[3]*sc->in->W->dims[2]*sc->in->W->dims[1] + d*sc->in->W->dims[3]*sc->in->W->dims[2] + n*h_kernel + m; // change h_kernel by W->dims[x]
                    value += val * sc->in->W->float_data[index_kernel];
                    //TRACE_LEVEL0("%fx%f+\n", val, sc->in->W->float_data[index_kernel]);
                  }
                }
              }
              sc->out->Y->float_data[out_index] = value;

              /* TODO This is a huge crap to make it work with tinyYOLO
              It adds the bias, but this if will waste a lot of time. Make
              this nice!
              */
              if (sc->in->B != NULL){
                sc->out->Y->float_data[out_index] += sc->in->B->float_data[k];
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

  debug_print_dims(sc->out->Y->n_dims, sc->out->Y->dims);
  return 0;
}

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
int operator_maxpool(operator__context *context)
{
  TRACE_LEVEL0("Calling operator_maxpool\n");
  //debug_print_attributes(n_attribute, attribute);

  operator__onnx__maxpool__context *sc = (operator__onnx__maxpool__context *) context;

  // number of dimensions do not change
  sc->out->Y->dims   = malloc(sc->in->X->n_dims * sizeof(int64_t));
  sc->out->Y->n_dims = sc->in->X->n_dims;

  int64_t h_kernel, w_kernel, h_stride, w_stride;
  h_kernel = w_kernel = h_stride = w_stride = 1;
  if (sc->attr->kernel_shape != NULL) {
    h_kernel = sc->attr->kernel_shape->ints[0];
    w_kernel = sc->attr->kernel_shape->ints[1];
  }

  if (sc->attr->strides != NULL) {
    h_stride = sc->attr->strides->ints[0];
    w_stride = sc->attr->strides->ints[1];
  }

  // TODO Maybe use a smaller type
  /* left and right pads */
  int h_pad, w_pad;
  h_pad = w_pad = 0;

  int h_pad_aux = 0;
  int w_pad_aux = 0;
  if (sc->attr->auto_pad != NULL){
    h_pad_aux = (h_kernel - 1);
    w_pad_aux = (w_kernel - 1);
    h_pad = (h_kernel - 1)/2;
    w_pad = (w_kernel - 1)/2;
    if (!strncmp((char*)sc->attr->auto_pad->s.data, "SAME_UPPER", 10)){
      // remove
    } else if (!strncmp((char*)sc->attr->auto_pad->s.data, "SAME_LOWER", 10)){
      /* TODO quick n dirty*/
      if ((h_kernel - 1)%2 != 0){
        h_pad++;
      }
      if ((w_kernel - 1)%2 != 0){
        w_pad++;
      }
    }
  }

  if (sc->attr->pads != NULL){
    /* TODO */
    /* Hardcoded for sc->attr->pads = [x, x, x, x] dim = 4*/
    h_pad_aux = sc->attr->pads->ints[0] + sc->attr->pads->ints[2];
    w_pad_aux = sc->attr->pads->ints[1] + sc->attr->pads->ints[3];
    h_pad = h_pad_aux/2;
    w_pad = w_pad_aux/2;

    /*
    for (int i = 0; i < pads->n_ints; i++){
      TRACE_LEVEL0("\n\n pad=%d\n", pads->ints[i]);
    }*/
  }

  sc->out->Y->dims[0] = sc->in->X->dims[0];
  sc->out->Y->dims[1] = sc->in->X->dims[1];
  sc->out->Y->dims[2] = (int64_t)floorf((float)(sc->in->X->dims[2] + h_pad_aux - ((h_kernel - 1) + 1)) / (float)h_stride + 1);
  sc->out->Y->dims[3] = (int64_t)floorf((float)(sc->in->X->dims[3] + w_pad_aux - ((w_kernel - 1) + 1)) / (float)w_stride + 1);

  sc->out->Y->has_raw_data = 0;

  switch(sc->in->X->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {
      sc->out->Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
      sc->out->Y->float_data = malloc(sc->out->Y->dims[0]*sc->out->Y->dims[1]*sc->out->Y->dims[2]*sc->out->Y->dims[3] * sizeof(float));
      sc->out->Y->n_float_data = sc->out->Y->dims[0]*sc->out->Y->dims[1]*sc->out->Y->dims[2]*sc->out->Y->dims[3];

      int b,i,j,k,m,n;
      for(b = 0; b < sc->out->Y->dims[0]; ++b){
        for(k = 0; k < sc->out->Y->dims[1]; ++k){
          for(i = 0; i < sc->out->Y->dims[2]; ++i){
            for(j = 0; j < sc->out->Y->dims[3]; ++j){
              int out_index = j + sc->out->Y->dims[3]*(i + sc->out->Y->dims[2]*(k + sc->in->X->dims[1]*b));
              float max = -999999; // TODO
              for(n = 0; n < h_kernel; ++n){
                for(m = 0; m < w_kernel; ++m){
                  int cur_h = i*h_stride + n -h_pad;
                  int cur_w = j*w_stride + m -w_pad;
                  int index = cur_w + sc->in->X->dims[3]*(cur_h + sc->in->X->dims[2]*(k + b*sc->in->X->dims[1]));
                  int valid = (cur_h >= 0 && cur_h < (sc->in->X->dims[2]) &&
                               cur_w >= 0 && cur_w < (sc->in->X->dims[3]));
                  float val = (valid != 0) ? sc->in->X->float_data[index] : -999999; //TODO
                  max = (val > max ? val : max);
                  }
                }
                sc->out->Y->float_data[out_index] = max;
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

  debug_print_dims(sc->out->Y->n_dims, sc->out->Y->dims);
  return 0;

}

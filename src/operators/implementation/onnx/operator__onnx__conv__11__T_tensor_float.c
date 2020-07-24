#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "trace.h"
#include "utils.h"
#include "tracing.h"
#include "index.h"

operator_status operator__onnx__conv__11__T_tensor_float(
    node_context *ctx
)
{
  TRACE_ENTRY(1);

  TRACE_NODE(2, true, ctx->onnx_node);

  Onnx__TensorProto *t_X = searchInputByName(ctx, 0);
  Onnx__TensorProto *t_W = searchInputByName(ctx, 1);
  Onnx__TensorProto *t_B = searchInputByName(ctx, 2);

  Onnx__TensorProto *t_Y = searchOutputByName(ctx, 0);

  TRACE_TENSOR(2, true, t_X);
  TRACE_TENSOR(2, true, t_W);
  TRACE(1, !t_B, "no bias given")
  TRACE_TENSOR(2, t_B, t_B);

  // X : (B x C x D1 x D2 x ... )
  int64_t B   = t_X->dims[0];
  int64_t C   = t_X->dims[1];
  int64_t n_D = t_X->n_dims-2;
  int64_t *D  = &t_X->dims[2];

  //TODO is the case of t_X->n_dims < 4 even valid for this operator?
  //TODO or is this a dirty hack for one of the models? :D
  if (t_X->n_dims == 2) {
    B   = 1;
    C   = 1;
    n_D = t_X->n_dims;
    D   = t_X->dims;
  }

  // W : (M x C/group x K1 x K2 x ... )
  int64_t M  = t_W->dims[0];
  int64_t group = (C + t_W->dims[1] -1) / t_W->dims[1];
  int64_t n_K = t_W->n_dims-2;
  __attribute__((unused))
  int64_t *K  = &t_W->dims[2];

  TRACE_FATAL(0, t_B && t_B->n_dims != 1 && t_B->dims[0] != M, "Mismatch of supplied bias size (%" PRId64 ") and number of feature maps (%" PRId64 ")", t_B->n_float_data, M);

  Onnx__AttributeProto *a_auto_pad     = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "auto_pad");
  Onnx__AttributeProto *a_dilations    = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "dilations");
  Onnx__AttributeProto *a_group        = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "group");
  Onnx__AttributeProto *a_kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "kernel_shape");
  Onnx__AttributeProto *a_pads         = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "pads");
  Onnx__AttributeProto *a_strides      = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "strides");

  TRACE_ATTRIBUTE(2, a_auto_pad, a_auto_pad);
  TRACE_ATTRIBUTE(2, a_dilations, a_dilations);
  TRACE_ATTRIBUTE(2, a_group, a_group);
  TRACE_ATTRIBUTE(2, a_kernel_shape, a_kernel_shape);
  TRACE_ATTRIBUTE(2, a_pads, a_pads);
  TRACE_ATTRIBUTE(2, a_strides, a_strides);

  //TODO clean up this string copy nonesense
  int n_auto_pad = (a_auto_pad?a_auto_pad->s.len:sizeof("NOTSET"))+1;
  char auto_pad[n_auto_pad];
  if (a_auto_pad) {
    strncpy(auto_pad,(char*)a_auto_pad->s.data,a_auto_pad->s.len);
  } else {
    strcpy(auto_pad,"NOTSET");
  }
  auto_pad[n_auto_pad-1] = '\0';

  TRACE_VAR(3, true, auto_pad, "\"%s\"");

  TRACE(1, !a_kernel_shape, "kernel dimensions are inferred from tensor W");
  __attribute__((unused))
  int64_t n_kernel = a_kernel_shape?a_kernel_shape->n_ints:t_W->n_dims;
  int64_t *kernel = a_kernel_shape?a_kernel_shape->ints:t_W->dims;
  TRACE_ARRAY(2, true, kernel, , n_kernel, "%" PRId64);

  //TODO check kernel dimensions against W and X dimensions
  //TODO aka see if kernel can be applied

  TRACE(1, !a_strides, "strides default to all 1")
  int64_t n_strides = a_strides?a_strides->n_ints:t_X->n_dims;
  int64_t strides[n_strides];
  for (int i = 0; i < n_strides; i++) {
    strides[i] = a_strides?a_strides->ints[i]:1;
  }
  TRACE_ARRAY(2, true, strides, , n_strides, "%" PRId64);

  TRACE(1, !a_pads && !a_auto_pad, "pads default to all 0")
  int64_t n_pads = n_D;
  int64_t pads_begin[n_pads];
  int64_t pads_end[n_pads];
  TRACE_WARN(0, a_pads && a_pads->n_ints != 2*n_pads, "dimension mismatch of provided paddings (%zu) and needed (%" PRId64 "), ignoring excess, applying auto_pad or filling with zeroes!", a_pads->n_ints, 2*n_pads);
  for(int i = 0; i < n_pads; i++) {
    if (a_pads && i < a_pads->n_ints/2) {
      pads_begin[i] = a_pads->ints[i];
      pads_end[i]   = a_pads->ints[a_pads->n_ints/2 + i];
      TRACE_WARN(0, strcmp(auto_pad, "NOTSET") != 0, "auto_pad and explicit padding specified for dimension %d: using explicit", i+2);
      continue;
    }
    if (strcmp(auto_pad, "VALID") == 0) {
      pads_begin[i] = 0;
      pads_end[i]   = 0;
      continue;
    }

    //reverse dimension calculation, see how t_Y->dims are calculated
    int64_t unstrided = D[i] * strides[i];
    int64_t input     = unstrided + (kernel[i] - 1);
    int64_t pads      = input - D[i];
    if (strcmp(auto_pad, "SAME_UPPER") == 0) {
      pads_begin[i] = pads / 2;
      pads_end[i]   = pads - pads_begin[i];
      continue;
    }
    if (strcmp(auto_pad, "SAME_LOWER") == 0) {
      pads_end[i]   = pads / 2;
      pads_begin[i] = pads - pads_end[i];
      continue;
    }
    TRACE_WARN(0, strcmp(auto_pad, "NOTSET") != 0, "no valid padding specified for dimension %d: zeroing padding", i+2);
    pads_begin[i] = 0;
    pads_end[i]   = 0;
  }
  TRACE_ARRAY(2, true, pads_begin, , n_pads, "%" PRId64);
  TRACE_ARRAY(2, true, pads_end, , n_pads, "%" PRId64);

  TRACE(1, !a_dilations, "dilations default to all 1")
  int64_t n_dilations = n_D;
  int64_t dilations[n_dilations];
  TRACE_WARN(0, a_dilations && a_dilations->n_ints != n_dilations, "dimension mismatch of provided dilations (%zu) and needed (%" PRId64 "), ignoring excess or filling up with ones!", a_dilations->n_ints, n_dilations);
  for(int i = 0; i < n_dilations; i++) {
    dilations[i] = a_dilations?a_dilations->ints[i]:1;
  }
  TRACE_ARRAY(2, true, dilations, , n_dilations, "%" PRId64);

  TRACE(1, !a_group, "group defaults to 1")
  __attribute__((unused))
  int64_t _group = a_group?a_group->i:1;
  TRACE_VAR(2, true, group, "%" PRId64);

  //TODO validate this check
  TRACE_FATAL(0, group != _group, "mismatch of supplied group (%" PRId64 ") and calculated group (%" PRId64 ")", _group, group);

  t_Y->n_dims = t_W->n_dims;
  t_Y->dims   = malloc(t_Y->n_dims * sizeof(int64_t));

  t_Y->dims[0] = B;
  t_Y->dims[1] = M;
  for(int i = 0; i < n_D; i++) {
    int64_t padded  = pads_begin[i] + D[i] + pads_end[i];
    int64_t output  = padded - (kernel[i] - 1);
    int64_t strided = (output + strides[i] -1) / strides[i];
    t_Y->dims[2+i] = strided;
  }

  t_Y->has_raw_data = 0;
  t_Y->data_type    = t_X->data_type;
  t_Y->n_float_data = 1;
  for (int i = 0; i < t_Y->n_dims; i++) {
    t_Y->n_float_data *= t_Y->dims[i];
  }
  t_Y->float_data   = malloc(t_Y->n_float_data * sizeof(float));

  TRACE_TENSOR(2, true, t_Y);

  /** general idea
   * - walk through all indices of the output
   * - calculate the output value of the output index by
   * - calculate the kernel & input indices needed
  **/

  if (t_X->n_dims == 4 || t_X->n_dims == 2) {
    //legacy code, almost untouched, fast for 2 and 4 dimension
    //TODO replace, refactor
    int64_t h_kernel = kernel[0];
    int64_t w_kernel = kernel[1];
    __attribute__((unused))
    int64_t d_kernel = kernel[2];
    int64_t h_stride = strides[0];
    int64_t w_stride = strides[1];
    int64_t h_dilation = dilations[0];
    int64_t w_dilation = dilations[1];
    int64_t h_pad    = pads_begin[0];
    int64_t w_pad    = pads_begin[1];
    int b, i, j, k, m, n, d;
    for(b = 0; b < t_Y->dims[0]; ++b){
      for(k = 0; k < t_Y->dims[1]; ++k){
        int g = (k/(t_Y->dims[1]/group));
        TRACE_BOUND_FATAL(3, true, g, 0, (int)group, "%d");
        for(i = 0; i < t_Y->dims[2]; ++i){
          for(j = 0; j < t_Y->dims[3]; ++j){
            // TODO replace all this calculations by macros?
            uint64_t out_index = j + t_Y->dims[3]*(i + t_Y->dims[2]*(k + t_Y->dims[1]*b));
            float value = 0;

            if (t_X->n_dims == 4){
              const int offset_in = g*(t_X->dims[3]*t_X->dims[2]*(t_X->dims[1]/group));
              TRACE_VAR(4, true, offset_in, "%d");
              for(d = 0; d < t_W->dims[1]; ++d){
                for(n = 0; n < h_kernel; ++n){   // TODO use t_W->dims[2] instead?
                  for(m = 0; m < w_kernel; ++m){ // TODO use t_W->dims[3] instead?
                    int cur_h = i*h_stride + n * h_dilation - h_pad;
                    int cur_w = j*w_stride + m * w_dilation - w_pad;

                    /* This is hardcoded to make it work with mnist model, where
                    the input is 1x1x28x28 */
                    int index = offset_in + cur_w + t_X->dims[3]*(cur_h + t_X->dims[2]*(d + 0*t_X->dims[1]));
                    //TRACE_LEVEL0("%d, %d, %d index=%d\n", d, cur_h, cur_w, index);

                    int valid = (cur_h >= 0 && cur_h < t_X->dims[2] &&
                                 cur_w >= 0 && cur_w < t_X->dims[3]);
                    TRACE_BOUND_FATAL(5, valid, index, 0, (int)t_X->n_float_data, "%d");
                    float val = (valid != 0) ? t_X->float_data[index] : 0;
                    int index_kernel = k*t_W->dims[3]*t_W->dims[2]*t_W->dims[1] + d*t_W->dims[3]*t_W->dims[2] + n*h_kernel + m; // change h_kernel by t_W->dims[x]
                    value += val * t_W->float_data[index_kernel];
                    //TRACE_LEVEL0("%fx%f+\n", val, t_W->float_data[index_kernel]);
                  }
                }
              }
            }else if (t_X->n_dims == 2){
              const int offset_in = g*(t_X->dims[1]/group);
              TRACE_VAR(4, true, offset_in, "%d");
              for(n = 0; n < h_kernel; ++n){
                for(m = 0; m < w_kernel; ++m){
                  int cur_h = i*h_stride + n * h_dilation - h_pad;
                  int cur_w = j*w_stride + m * w_dilation - w_pad;
                  //printf("%d, %d\n", cur_h, cur_w);
                  int index = offset_in + cur_w + t_X->dims[1]*cur_h;
                  //printf("index=%d\n", index);
                  int valid = (cur_h >= 0 && cur_h < t_X->dims[0] &&
                               cur_w >= 0 && cur_w < t_X->dims[1]);
                  TRACE_BOUND_FATAL(5, valid, index, 0, (int)t_X->n_float_data, "%d");
                  float val = (valid != 0) ? t_X->float_data[index] : 0;
                  int index_kernel = k*t_W->dims[3]*t_W->dims[2]*t_W->dims[1] + n*h_kernel + m;
                  value += val * t_W->float_data[index_kernel];
                  //TRACE_LEVEL0("%fx%f+\n", val, input[1]->float_data[index_kernel]);
                }
              }
            }else{
              /* TODO */
            }
            TRACE_BOUND_FATAL(4, true, (size_t)out_index, (size_t)0, t_Y->n_float_data, "%zu");
            t_Y->float_data[out_index] = value;
            //printf("%lld\n", out_index);
            //printf("[%lld]=%f\n", out_index, value);

            /* TODO This is a huge crap to make it work with tinyYOLO
            It adds the bias, but this if will waste a lot of time. Make
            this nice!
            */
            if (t_B != NULL){
              t_Y->float_data[out_index] += t_B->float_data[k];
            }
          }
        }
      }
    }
    TRACE_EXIT(1);
    return 0;
  }

  TRACE_WARN(0, true, "running generic implementation...slow!");

  index_ctx kernel_index;
  int64_t   kernel_indices[t_W->n_dims];
  int64_t   kernel_offsets[t_W->n_dims];
  index_init(&kernel_index, t_W->n_dims, t_W->dims, kernel_indices, kernel_offsets);

  index_ctx input_index;
  int64_t   input_indices[t_X->n_dims];
  int64_t   input_offsets[t_X->n_dims];
  index_init(&input_index, t_X->n_dims, t_X->dims, input_indices, input_offsets);

  index_ctx output_index;
  int64_t   output_indices[t_Y->n_dims];
  int64_t   output_offsets[t_Y->n_dims];
  index_init(&output_index, t_Y->n_dims, t_Y->dims, output_indices, output_offsets);

  do {
    TRACE_INDEX(3, true, &output_index);
    float value = t_B?t_B->float_data[index_get(&output_index, 1)]:0;
    int64_t input_ch_offset = index_get(&output_index, 1)/(M/group);
    TRACE_VAR(3, true, input_ch_offset, "%" PRId64);
    index_set(&kernel_index, 0, index_get(&output_index, 1));
    index_set(&input_index, 0, index_get(&output_index, 0));
    for (int64_t ch = 0; ch < C/group; ch++) {
      TRACE_BOUND(4, true, ch, (int64_t)0, C, "%" PRId64);
      index_set(&kernel_index, 1, ch);
      index_set(&input_index, 1, ch + input_ch_offset);
      index_reset_sub(&kernel_index, 2);
      do {
        TRACE_INDEX(5, true, &kernel_index);
        bool is_padded = false;
        for (int i = 0; i < n_K; i++) {
          int64_t input = 0;
          input += index_get(&output_index, 2+i) * strides[i];
          input += index_get(&kernel_index, 2+i) * dilations[i];
          TRACE(6, true, "dim %d index %" PRId64 " valid in %" PRId64 ":%" PRId64 , 2+i, input, pads_begin[i], t_X->dims[2+i]);
          if ( input < pads_begin[i] || input - pads_begin[i] >= t_X->dims[2+i]  ) {
            is_padded = true;
            TRACE(6, true, "dim %d index %" PRId64 " is padded, skipping", 2+i, input);
            break;
          }
          index_set(&input_index, 2+i, input - pads_begin[i]);
        }
        if (is_padded) {
          continue;
        }
        TRACE_INDEX(5, true, &input_index);
        float   data_input = t_X->float_data[input_index.offset];
        float   data_kernel = t_W->float_data[kernel_index.offset];
        TRACE_VAR(5, true, data_input, "%f");
        TRACE_VAR(5, true, data_kernel, "%f");
        value +=  data_input * data_kernel;
        TRACE_VAR(5, true, value, "%f");
      } while(index_inc_sub(&kernel_index, 1));
    }
    TRACE(3, true, "writing value: %f @ %" PRId64, value, output_index.offset);
    t_Y->float_data[output_index.offset] = value;
  } while (index_inc(&output_index));

  TRACE_EXIT(1);
  return 0;
}

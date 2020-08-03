#include "operators/ai.onnx/Softmax/11/operator__ai_onnx__softmax__11.h"

#include "tracing.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>

static inline
void
softmax(float *in, float *out, int num) {

    float max = 0;
    for (int i = 0; i < num; i++) {
        if (in[i] > max) {
            max = in[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < num; i++) {
        //reducing all arguments by fixed value (max) keeps ratio,
        //but expf will never return inf for large arguments
        float e = expf(in[i] - max);
        sum += e;
        out[i] = e;
    }
    for (int i = 0; i < num; i++) {
        out[i] /= sum;
    }
}

operator_status
operator__ai_onnx__softmax__11__T_tensor_float(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    Onnx__TensorProto *t_input  = searchInputByName(ctx, 0);
    Onnx__TensorProto *t_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");

    TRACE_TENSOR(2,true,t_input);
    TRACE_ATTRIBUTE(2, a_axis, a_axis);

    //wrap axis if negative
    int axis = a_axis?a_axis->i:1;
    if (axis < 0) {
        axis += t_input->n_dims;
    }

    t_output->has_raw_data = 0;
    t_output->data_type = t_input->data_type;
    t_output->n_dims = t_input->n_dims;
    t_output->dims = malloc(t_output->n_dims * sizeof(int64_t));
    t_output->n_float_data = t_input->n_float_data;
    t_output->float_data = malloc(t_output->n_float_data * sizeof(float));

    int N = 1;
    int D = 1;
    for (int i = 0; i < t_input->n_dims; i++) {
        if (i < axis) {
            N *= t_input->dims[i];
        } else {
            D *= t_input->dims[i];
        }
        t_output->dims[i] = t_input->dims[i];
    }

    for (int n = 0; n < N; n++) {
        int offset = D*n;
        softmax(&t_input->float_data[offset], &t_output->float_data[offset], D);
    }

    TRACE_TENSOR(2,true,t_output);
    TRACE_EXIT(1);
    return OP_OK;
}
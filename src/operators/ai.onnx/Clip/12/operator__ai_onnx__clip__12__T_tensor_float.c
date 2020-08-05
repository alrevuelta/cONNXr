#include "operators/ai.onnx/Clip/12/operator__ai_onnx__clip__12.h"

#include "tracing.h"
#include "utils.h"
#include <float.h>
#include <stdlib.h>

operator_status
operator__ai_onnx__clip__12__T_tensor_float(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    Onnx__TensorProto *t_input  = searchInputByName(ctx, 0);
    Onnx__TensorProto *t_min    = searchInputByName(ctx, 1);
    Onnx__TensorProto *t_max    = searchInputByName(ctx, 2);

    TRACE_TENSOR(2, true, t_input);
    TRACE_TENSOR(2, t_min, t_min);
    TRACE_TENSOR(2, t_max, t_max);

    float min = t_min?t_min->float_data[0]:-FLT_MAX;
    float max = t_max?t_max->float_data[0]:FLT_MAX;

    Onnx__TensorProto *t_output = searchOutputByName(ctx, 0);

    t_output->has_raw_data = 0;
    t_output->data_type = t_input->data_type;
    t_output->n_float_data = t_input->n_float_data;
    t_output->n_dims = t_input->n_dims;
    t_output->dims = malloc(t_output->n_dims * sizeof(int64_t));
    for (int i = 0; i < t_output->n_dims; i++) {
      t_output->dims[i] = t_input->dims[i];
    }

    t_output->float_data = malloc(t_output->n_float_data * sizeof(float));
    for(int i = 0; i < t_input->n_float_data; i++) {
        float input = t_input->float_data[i];
        float *output = &t_output->float_data[i];
        if (input > max) {
            *output = max;
            continue;
        }
        if (input < min) {
            *output = min;
            continue;
        }
        *output = input;
    }

    TRACE_TENSOR(2, true, t_output);
    TRACE_EXIT(1);

    return OP_OK;
}
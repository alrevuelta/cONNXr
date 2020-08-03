#include "operators/ai.onnx/GlobalAveragePool/1/operator__ai_onnx__globalaveragepool__1.h"

#include "tracing.h"
#include "utils.h"
#include <stdlib.h>

operator_status
operator__ai_onnx__globalaveragepool__1__T_tensor_float(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    Onnx__TensorProto *t_input  = searchInputByName(ctx, 0);

    TRACE_TENSOR(2, true, t_input);

    Onnx__TensorProto *t_output = searchOutputByName(ctx, 0);

    t_output->has_raw_data = 0;
    t_output->data_type = t_input->data_type;
    t_output->n_dims = t_input->n_dims;
    t_output->dims = malloc(t_output->n_dims * sizeof(int64_t));

    // assuming we have at least 3 dimensions ( N x C x D1 x D2 ... )

    // N x C
    t_output->n_float_data = 1;
    for (int i = 0; i < 2; i++) {
      t_output->dims[i] = t_input->dims[i];
      t_output->n_float_data *= t_input->dims[i];
    }
    // D1 x D2 x ...
    int64_t cardinality = 1;
    for (int i = 2; i < t_input->n_dims; i++) {
      t_output->dims[i] = 1;
      cardinality *= t_input->dims[i];
    }

    t_output->float_data = malloc(t_output->n_float_data * sizeof(float));
    for (int n = 0; n < t_input->dims[0]; n++) {
        for (int c = 0; c < t_input->dims[1]; c++) {
            int offset = n*t_input->dims[1]*cardinality + c*cardinality;
            float sum = 0;
            for (int i = 0; i < cardinality; i++) {
                sum += t_input->float_data[offset+i];
            }
            t_output->float_data[n*t_input->dims[1] + c] = sum / cardinality;
        }
    }

    TRACE_TENSOR(2, true, t_output);
    TRACE_EXIT(1);

    return OP_OK;
}
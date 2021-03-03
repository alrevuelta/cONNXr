//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__identity__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>

operator_status
prepare_operator__ai_onnx__identity__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);

    TRACE_TENSOR(2, true, i_input);

    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__identity__1 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__identity__1));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */

    char *saved_name = o_output->name;
    memcpy(o_output,i_input,sizeof(Onnx__TensorProto));
    o_output->name = saved_name;

    /* MALLOC OUTPUT TENSORS HERE */

    // mallocTensorData(o_output);

    TRACE_TENSOR(2, true, o_output);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = &execute_operator__ai_onnx__identity__1;
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    // return OP_ENOSYS;
    return OP_OK;
}
//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__convtranspose__11.h"
#include "tracing.h"
#include "utils.h"

int calcOutputSize(int inputSize, int kernelSize, int stride, int dilations, int padStart, int padEnd) {
    return stride * (inputSize - 1) + ((kernelSize - 1) * dilations + 1 - padStart - padEnd);
}

int64_t* createArray(size_t n, int64_t value) {
    int64_t* p = (int64_t*)malloc(n * sizeof(int64_t));
    for(int i = 0; i < n; i++) {
        p[i] = value;
    }
    return p;
}

operator_status
prepare_operator__ai_onnx__convtranspose__11(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_W = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 2);

    TRACE_TENSOR(2, true, i_X);
    TRACE_TENSOR(2, true, i_W);
    TRACE_TENSOR(2, B, i_B);

    // Onnx__AttributeProto *a_auto_pad = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"auto_pad");
    Onnx__AttributeProto *a_dilations = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"dilations");
    Onnx__AttributeProto *a_group = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"group");
    //https://github.com/onnx/onnx/issues/785
    //'kernel_shape' parameter is redundant and can be inferred from the input
    // Onnx__AttributeProto *a_kernel_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"kernel_shape");
    // Onnx__AttributeProto *a_output_padding = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"output_padding");
    // Onnx__AttributeProto *a_output_shape = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"output_shape");
    Onnx__AttributeProto *a_pads = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"pads");
    Onnx__AttributeProto *a_strides = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"strides");

    // TRACE_ATTRIBUTE(2, a_auto_pad, a_auto_pad);
    TRACE_ATTRIBUTE(2, a_dilations, a_dilations);
    TRACE_ATTRIBUTE(2, a_group, a_group);
    // TRACE_ATTRIBUTE(2, a_kernel_shape, a_kernel_shape);
    // TRACE_ATTRIBUTE(2, a_output_padding, a_output_padding);
    // TRACE_ATTRIBUTE(2, a_output_shape, a_output_shape);
    TRACE_ATTRIBUTE(2, a_pads, a_pads);
    TRACE_ATTRIBUTE(2, a_strides, a_strides);

    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    TRACE_FATAL(0, i_X->n_dims != 4, "use convtranspose only with a 2D input (4D tensor), %zu was given", i_X->n_dims);
    TRACE_FATAL(0, i_X->dims[0]!= 1, "use convtranspose only with batchsize of 1, %zu was given", i_X->dims[0]);

    int inputSizeX = i_X->dims[3];
    int inputSizeY = i_X->dims[2];
    //int inputChannels = i_X->dims[1];

    TRACE_FATAL(0, i_X->dims[1] != i_W->dims[0], "size of input channels needs to be qual with weights");
    
    int kernelSizeX = i_W->dims[3];
    int kernelSizeY = i_W->dims[2];
    int outputChannels = i_W->dims[1];

    if(i_B != NULL) {
        //int biasSize = i_B->dims[0];
        TRACE_FATAL(0, i_B->dims[0] != i_W->dims[1] * default_group, "size of input channels needs to be qual with bias");
    }

    //number of tensor axes minus batch minus channel 
    int dim = i_X->n_dims - 2;
    
    int64_t default_group = 1;
    // char* default_auto_pad = ;
    size_t default_n_dilations = dim;
    int64_t default_dilation = 1;
    // int64_t* default_dilations = ;
    // size_t default_n_kernel_shape = ;
    // int64_t* default_kernel_shape = ;
    // size_t default_n_output_padding = ;
    // int64_t* default_output_padding = ;
    // size_t default_n_output_shape = ;
    // int64_t* default_output_shape = ;
    size_t default_n_pads = 2 * dim;
    int64_t default_pad = 0;
    // int64_t* default_pads = ;
    size_t default_n_strides = dim;
    int64_t default_stride = 1;
    // int64_t* default_strides = ;

    context_operator__ai_onnx__convtranspose__11 *op_ctx = NULL;
    op_ctx = malloc(sizeof(context_operator__ai_onnx__convtranspose__11));
    TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    // op_ctx->auto_pad = a_auto_pad?strndup((char*)a_auto_pad->s.data, a_auto_pad->s.len):default_auto_pad;
    op_ctx->n_dilations = a_dilations?a_dilations->n_ints:default_n_dilations;
    op_ctx->dilations = a_dilations?ARRAYDUP(a_dilations->ints,default_n_dilations):createArray(default_n_dilations, default_dilation);
    TRACE_FATAL(0, !op_ctx->dilations, "malloc failed");
    op_ctx->group = a_group?a_group->i:default_group;

    TRACE_FATAL(0, op_ctx->group != 1, "use convtranspose only with a groupsize of 1, %zu was given", op_ctx->group);

    // op_ctx->n_kernel_shape = a_kernel_shape?a_kernel_shape->n_ints:default_n_kernel_shape;
    // op_ctx->kernel_shape = a_kernel_shape?a_kernel_shape->ints:ARRAYDUP(default_kernel_shape,default_n_kernel_shape);
    // TRACE_FATAL(0, !op_ctx->kernel_shape, "malloc failed");
    // op_ctx->n_output_padding = a_output_padding?a_output_padding->n_ints:default_n_output_padding;
    // op_ctx->output_padding = a_output_padding?a_output_padding->ints:ARRAYDUP(default_output_padding,default_n_output_padding);
    // TRACE_FATAL(0, !op_ctx->output_padding, "malloc failed");
    // op_ctx->n_output_shape = a_output_shape?a_output_shape->n_ints:default_n_output_shape;
    // op_ctx->output_shape = a_output_shape?a_output_shape->ints:ARRAYDUP(default_output_shape,default_n_output_shape);
    // TRACE_FATAL(0, !op_ctx->output_shape, "malloc failed");
    op_ctx->n_pads = a_pads?a_pads->n_ints:default_n_pads;
    op_ctx->pads = a_pads?ARRAYDUP(a_pads->ints,op_ctx->n_pads):createArray(default_n_pads, default_pad);
    TRACE_FATAL(0, !op_ctx->pads, "malloc failed");
    op_ctx->n_strides = a_strides?a_strides->n_ints:default_n_strides;
    op_ctx->strides = a_strides?ARRAYDUP(a_strides->ints,default_n_strides):createArray(default_n_strides, default_stride);
    TRACE_FATAL(0, !op_ctx->strides, "malloc failed");

    // TRACE_VAR(2, true, op_ctx->auto_pad, "\"%s\"");
    TRACE_ARRAY(2, true, op_ctx->dilations, , op_ctx->n_dilations, "%" PRId64);
    // TRACE_VAR(2, true, op_ctx->group, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->kernel_shape, , op_ctx->n_kernel_shape, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->output_padding, , op_ctx->n_output_padding, "%" PRId64);
    // TRACE_ARRAY(2, true, op_ctx->output_shape, , op_ctx->n_output_shape, "%" PRId64);
    TRACE_ARRAY(2, true, op_ctx->pads, , op_ctx->n_pads, "%" PRId64);
    TRACE_ARRAY(2, true, op_ctx->strides, , op_ctx->n_strides, "%" PRId64);

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */

    const int strideX = op_ctx->strides[1];
    const int strideY = op_ctx->strides[0];

    const int dilationsX = op_ctx->dilations[1];
    const int dilationsY = op_ctx->dilations[0];

    const int padStartY = op_ctx->pads[0];
    const int padStartX = op_ctx->pads[1];
    const int padEndY = op_ctx->pads[2];
    const int padEndX = op_ctx->pads[3];

    int outputSizeX = calcOutputSize(inputSizeX, kernelSizeX, strideX, dilationsX, padStartX, padEndX);
    int outputSizeY = calcOutputSize(inputSizeY, kernelSizeY, strideY, dilationsY, padStartY, padEndY);

    o_Y->n_dims       = i_X->n_dims;
    o_Y->dims         = malloc(o_Y->n_dims * sizeof(int64_t));
    TRACE_FATAL(0, !op_ctx->dilations, "malloc failed");
    o_Y->has_raw_data = 0;
    o_Y->data_type    = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
    o_Y->n_float_data = outputSizeX * outputSizeY * outputChannels;

    o_Y->dims[0] = 1;
    o_Y->dims[1] = outputChannels;
    o_Y->dims[2] = outputSizeY;
    o_Y->dims[3] = outputSizeX;

    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_Y);

    // TRACE_TENSOR(2, true, o_Y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = resolve_operator__ai_onnx__convtranspose__11(ctx);
    ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    //return OP_ENOSYS;
    return OP_OK;
}
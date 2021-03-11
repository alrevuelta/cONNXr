//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__convtranspose__11.h"
#include "tracing.h"
#include "utils.h"

//defined in prepare
int calcOutputSize(int inputSize, int kernelSize, int stride, int dilations, int padStart, int padEnd);

//transformes a 3d pos into a 1d flat array pos
static inline int calcArrayPos3D(int x, int y, int outputChannel, int width, int height) {
    return outputChannel * height * width + y * width + x;
}

//transformes a 4d pos into a 1d flat array pos
static inline int calcArrayPos4D(int x, int y, int outputChannel, int intputChannel, int width, int height, int nOfOutputChannels) {
    return intputChannel * nOfOutputChannels *height * width 
           + outputChannel * height * width 
           + y * width
           + x;
}

operator_status
execute_operator__ai_onnx__convtranspose__11__T_tensor_float(
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

    context_operator__ai_onnx__convtranspose__11 *op_ctx = ctx->executer_context;

    // size is not needed, because this operator works for one fixed size only
    
    // char* auto_pad = op_ctx->auto_pad;
    // size_t n_dilations = op_ctx->n_dilations;
    int64_t* dilations = op_ctx->dilations;
    // int64_t group = op_ctx->group;
    // size_t n_kernel_shape = op_ctx->n_kernel_shape;
    // int64_t* kernel_shape = op_ctx->kernel_shape;
    // size_t n_output_padding = op_ctx->n_output_padding;
    // int64_t* output_padding = op_ctx->output_padding;
    // size_t n_output_shape = op_ctx->n_output_shape;
    // int64_t* output_shape = op_ctx->output_shape;
    // size_t n_pads = op_ctx->n_pads;
    int64_t* pads = op_ctx->pads;
    // size_t n_strides = op_ctx->n_strides;
    int64_t* strides = op_ctx->strides;

    // TRACE_VAR(2, true, auto_pad, "\"%s\"");
    TRACE_ARRAY(2, true, dilations, , n_dilations, "%" PRId64);
    // TRACE_VAR(2, true, group, "%" PRId64);
    // TRACE_ARRAY(2, true, kernel_shape, , n_kernel_shape, "%" PRId64);
    // TRACE_ARRAY(2, true, output_padding, , n_output_padding, "%" PRId64);
    // TRACE_ARRAY(2, true, output_shape, , n_output_shape, "%" PRId64);
    TRACE_ARRAY(2, true, pads, , n_pads, "%" PRId64);
    TRACE_ARRAY(2, true, strides, , n_strides, "%" PRId64);

    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    printf("%p\n", o_Y->float_data);

    // TRACE_TENSOR(2, true, o_Y);

    /* DO CALCULATION HERE */

    const int inputSizeX = i_X->dims[3];
    const int inputSizeY = i_X->dims[2];
    const int inputChannels = i_X->dims[1];

    const float *input = i_X->float_data;
    const float *weights = i_W->float_data;

    const int kernelSizeX = i_W->dims[3];
    const int kernelSizeY = i_W->dims[2];
    const int outputChannels = i_W->dims[1];

    const int strideX = strides[1];
    const int strideY = strides[0];

    const int dilationsX = dilations[1];
    const int dilationsY = dilations[0];

    const int padStartY = pads[0];
    const int padStartX = pads[1];
    const int padEndY = pads[2];
    const int padEndX = pads[3];

    const int outputSizeX = calcOutputSize(inputSizeX, kernelSizeX, strideX, dilationsX, padStartX, padEndX);
    const int outputSizeY = calcOutputSize(inputSizeY, kernelSizeY, strideY, dilationsY, padStartY, padEndY);

    float* output = o_Y->float_data;

    //fill with bias
    for(int c=0; c<outputChannels; c++) {
        float bias = i_B?i_B->float_data[c]:0;
        for(int y=0; y<outputSizeY; y++) {
            for(int x=0; x<outputSizeX; x++) {
                output[calcArrayPos3D(x,y,c,outputSizeX, outputSizeY)] = bias;
            }
        }
    }

    //actual transpose convolution
    for(int i=0; i < inputChannels; i++) {
        for(int c=0; c<outputChannels; c++) {
            for(int inputPosY=0; inputPosY<inputSizeY; inputPosY++) {
                for(int inputPosX=0; inputPosX<inputSizeX; inputPosX++) {
                    float _input = input[calcArrayPos3D(inputPosX, inputPosY, i, inputSizeX, inputSizeY)];

                    for(int kernelPosX=0; kernelPosX<kernelSizeX; kernelPosX++) {
                        int x = inputPosX*strideX+kernelPosX*dilationsX - padStartX;
                        if(x < 0) {
                            continue;
                        } else if (x >= outputSizeX) {
                            continue;
                        }

                        for(int kernelPosY=0; kernelPosY<kernelSizeY; kernelPosY++) {
                            int y = inputPosY*strideY+kernelPosY*dilationsY - padStartY;

                            if(y < 0) {
                                continue;
                            } else if (y >= outputSizeY) {
                                continue;
                            }

                            const float _weight = weights[calcArrayPos4D(kernelPosX, kernelPosY, c, i, kernelSizeX, kernelSizeY, outputChannels)];
                            output[calcArrayPos3D(x, y, c, outputSizeX, outputSizeY)] += _input * _weight;
                        }
                    }
                }
            }
        }
    }

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS EXECUTER IS VALID */
    return OP_OK;
}

//this file was generated by ../../../scripts/onnx_generator/OperatorTypeResolver.py
#include "operators/onnx/operator__onnx__sigmoid__6.h"
#include "operators/operator_stub.h"
#include "operators.h"
#include <inttypes.h>
#include <stdio.h>

operator_executer resolve_operator__onnx__sigmoid__6(
    node_context *ctx
){
    operator_executer executer = NULL;
    {
    uint32_t T = 0;
    if (ctx->inputs[0]) {
        T = ctx->inputs[0]->data_type;
    }
    switch ( T ) {
        case 0: //constrained tensor is not set (maybe optional?), just take next case
        //case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: { executer = (operator_executer) &operator__onnx__sigmoid__6__T_tensor_double; break; }
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: { executer = (operator_executer) &operator__onnx__sigmoid__6__T_tensor_float; break; }
        //case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16: { executer = (operator_executer) &operator__onnx__sigmoid__6__T_tensor_float16; break; }
        default: {
            fprintf(stderr, "no matching type for constraint 'T' found!\n");
            break;
        }
    }
}
    if (!executer) {
        executer = &operator_stub;
    }
    return executer;
}

//this file was generated by ../../../../../scripts/onnx_generator/OperatorSets.py

#include "config.h"
#include "operators/operator_set.h"

extern operator_set_opversion opversion_operator__ai_onnx__relu__6;

operator_set_opname opname_operator__ai_onnx__relu = {
    .name = "Relu",
    .opversions = {
#ifdef CONFIG_HAVE_OPERATOR__AI_ONNX__RELU__6
        &opversion_operator__ai_onnx__relu__6,
#endif
        NULL
    }
};
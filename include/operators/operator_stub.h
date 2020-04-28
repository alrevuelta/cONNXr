#ifndef OPERATOR_STUB_H
#define OPERATOR_STUB_H

#include "operators/operator.h"

int operator_stub(
    size_t n_input,
    Onnx__TensorProto** input,
    size_t n_attribute,
    Onnx__AttributeProto** attribute,
    size_t n_output,
    Onnx__TensorProto** output
);

#endif
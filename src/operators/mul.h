#ifndef EMBEDDEDML_MUL_H
#define EMBEDDEDML_MUL_H
#include "../pb/onnx.pb-c.h"

int operator_mul(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output);

#endif

#ifndef EMBEDDEDML_CONV_H
#define EMBEDDEDML_CONV_H
#include "../pb/onnx.pb-c.h"

void operator_conv(size_t n_input,
                   Onnx__TensorProto **input,
                   size_t n_attribute,
                   Onnx__AttributeProto **attribute,
                   size_t n_output,
                   Onnx__TensorProto **output);

#endif

#ifndef EMBEDDEDML_CONV_H
#define EMBEDDEDML_CONV_H
#include "../pb/onnx.pb-c.h"

void operator_conv(Onnx__TensorProto *X,
                   Onnx__TensorProto *W,
                   Onnx__TensorProto *B,
                   Onnx__TensorProto *Y,
                   size_t n_attribute,
                   Onnx__AttributeProto **attribute);

#endif

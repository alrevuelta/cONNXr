#ifndef EMBEDDEDML_RELU_H
#define EMBEDDEDML_RELU_H
#include "../onnx.pb-c.h"

void operator_relu(Onnx__TensorProto *X, Onnx__TensorProto *Y);

#endif

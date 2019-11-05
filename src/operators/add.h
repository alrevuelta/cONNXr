#ifndef EMBEDDEDML_ADD_H
#define EMBEDDEDML_ADD_H
#include "../onnx.pb-c.h"

void operator_add(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c);

#endif

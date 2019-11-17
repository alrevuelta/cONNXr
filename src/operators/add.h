#ifndef EMBEDDEDML_ADD_H
#define EMBEDDEDML_ADD_H
#include "../pb/onnx.pb-c.h"

void operator_add(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *c);

#endif

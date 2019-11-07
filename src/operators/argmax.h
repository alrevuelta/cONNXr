#ifndef EMBEDDEDML_ARGMAX_H
#define EMBEDDEDML_ARGMAX_H
#include "../onnx.pb-c.h"

void operators_argmax(Onnx__TensorProto *data, int axis, int keepdims, Onnx__TensorProto *reduced);

#endif

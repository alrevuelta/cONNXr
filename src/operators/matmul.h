#ifndef EMBEDDEDML_MATMUL_H
#define EMBEDDEDML_MATMUL_H
#include "../onnx.pb-c.h"

void operator_matmul(Onnx__TensorProto *a, Onnx__TensorProto *b, Onnx__TensorProto *o);

#endif

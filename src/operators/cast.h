#ifndef EMBEDDEDML_CAST_H
#define EMBEDDEDML_CAST_H
#include "../onnx.pb-c.h"

void operators_cast(Onnx__TensorProto *T1, Onnx__TensorProto *T2, Onnx__TensorProto__DataType to);

#endif

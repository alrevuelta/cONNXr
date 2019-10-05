#ifndef EMBEDDEDML_OPWRAPPER_H
#define EMBEDDEDML_OPWRAPPER_H
#include "onnx.pb-c.h"

void Operators_MatMul(void *in, void *matrix, int m, int n, int k, void *out, enum _Onnx__TensorProto__DataType type);
void Operators_Add(void *inOut, void *matrix, int m, enum _Onnx__TensorProto__DataType type);

#endif

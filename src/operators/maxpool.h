#ifndef EMBEDDEDML_MAXPOOL_H
#define EMBEDDEDML_MAXPOOL_H
#include "../pb/onnx.pb-c.h"

void operator_maxpool(Onnx__TensorProto *X,
                      Onnx__TensorProto *Y,
                      Onnx__TensorProto *Indices,
                      size_t n_attribute,
                      Onnx__AttributeProto **attribute);



#endif

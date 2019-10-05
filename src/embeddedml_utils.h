#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "onnx.pb-c.h"

Onnx__TensorProto *searchTensorForNode(Onnx__ModelProto *model, int nodeIdx);
int getDimensionsOfTensor(Onnx__TensorProto *tensor);
Onnx__ModelProto *openOnnxFile(char *fname);

#endif

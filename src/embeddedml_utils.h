#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "onnx.pb-c.h"

Onnx__TensorProto *searchTensorInInitializers(Onnx__ModelProto *model, char *name);
int getDimensionsOfTensor(Onnx__TensorProto *tensor);
Onnx__ModelProto *openOnnxFile(char *fname);

#endif

#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "onnx.pb-c.h"

Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs, char *name);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

#endif

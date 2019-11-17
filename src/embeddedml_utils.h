#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "pb/onnx.pb-c.h"

Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs, char *name);
Onnx__AttributeProto* searchAttributeNyName(size_t n_attribute, Onnx__AttributeProto **attribute, char *name);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

#endif

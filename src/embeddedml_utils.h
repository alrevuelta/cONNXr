#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "pb/onnx.pb-c.h"

Onnx__TensorProto* searchTensorProtoByName(const Onnx__ModelProto *model,
                                           Onnx__TensorProto **inputs,
                                           const int nInputs,
                                           const char *name);
Onnx__AttributeProto* searchAttributeNyName(const size_t n_attribute,
                                            const Onnx__AttributeProto **attribute,
                                            const char *name);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

#endif

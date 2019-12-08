#ifndef EMBEDDEDML_UTILS_H
#define EMBEDDEDML_UTILS_H
#include "pb/onnx.pb-c.h"

/* Max size for strings */
#define MAX_CHAR_SIZE 50

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

void mallocTensorProto(Onnx__TensorProto *tp,
                       Onnx__TensorProto__DataType data_type,
                       size_t n_dims,
                       size_t n_data);

#endif

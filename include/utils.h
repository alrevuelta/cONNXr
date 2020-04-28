#ifndef UTILS_H
#define UTILS_H
#include "onnx.pb-c.h"

/* Max size for strings */
#define MAX_CHAR_SIZE 50

Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model,
                                           Onnx__TensorProto **inputs,
                                           int nInputs,
                                           char *name);
Onnx__AttributeProto* searchAttributeNyName(size_t n_attribute,
                                            Onnx__AttributeProto **attribute,
                                            char *name);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

void mallocTensorProto(Onnx__TensorProto *tp,
                       Onnx__TensorProto__DataType data_type,
                       size_t n_dims,
                       size_t n_data);

#endif

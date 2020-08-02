#ifndef UTILS_H
#define UTILS_H
#include "onnx.pb-c.h"
#include "inference.h"

/* Max size for strings */
#define MAX_CHAR_SIZE 50

Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model,
                                           Onnx__TensorProto **inputs,
                                           int nInputs,
                                           char *name);
Onnx__AttributeProto* searchAttributeNyName(size_t n_attribute,
                                            Onnx__AttributeProto **attribute,
                                            char *name);
Onnx__TensorProto* searchInputByName(node_context *ctx,
                                     int index);
Onnx__TensorProto* searchOutputByName(node_context *ctx,
                                      int index);
Onnx__ModelProto* openOnnxFile(char *fname);
Onnx__TensorProto* openTensorProtoFile(char *fname);

size_t exportTensorProtoFile(const Onnx__TensorProto *tensor, char *fname);

int convertRawDataOfTensorProto(Onnx__TensorProto *tensor);

void mallocTensorProto(Onnx__TensorProto *tp,
                       Onnx__TensorProto__DataType data_type,
                       size_t n_dims,
                       size_t n_data);

void init_tensor_proto(Onnx__TensorProto *tp);

size_t strnlen(const char *src, size_t length);
char*  strndup(const char *src, size_t length);
char*  strdup(const char *src);

#endif

#ifndef EMBEDDEDML_MATMUL_H
#define EMBEDDEDML_MATMUL_H
#include "../pb/onnx.pb-c.h"

int operator_matmul(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

#endif

#ifndef EMBEDDEDML_MATMUL_H
#define EMBEDDEDML_MATMUL_H
#include "../pb/onnx.pb-c.h"

int operator_matmul(const size_t n_input,
                    const Onnx__TensorProto **input,
                    const size_t n_attribute,
                    const Onnx__AttributeProto **attribute,
                    const size_t n_output,
                    Onnx__TensorProto **output);

#endif

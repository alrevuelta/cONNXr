#ifndef EMBEDDEDML_BATCHNORMALIZATION_H
#define EMBEDDEDML_BATCHNORMALIZATION_H
#include "../pb/onnx.pb-c.h"

int operator_batchnormalization(const size_t n_input,
                                const Onnx__TensorProto **input,
                                const size_t n_attribute,
                                const Onnx__AttributeProto **attribute,
                                const size_t n_output,
                                Onnx__TensorProto **output);

#endif

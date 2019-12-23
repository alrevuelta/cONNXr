#ifndef EMBEDDEDML_BATCHNORMALIZATION_H
#define EMBEDDEDML_BATCHNORMALIZATION_H
#include "../pb/onnx.pb-c.h"

int operator_batchnormalization(size_t n_input,
                                Onnx__TensorProto **input,
                                size_t n_attribute,
                                Onnx__AttributeProto **attribute,
                                size_t n_output,
                                Onnx__TensorProto **output);

#endif

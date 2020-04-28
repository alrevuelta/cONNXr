#ifndef OPERATORS_H
#define OPERATORS_H
#include "onnx.pb-c.h"

int operator_add(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output);

int operator_argmax(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_batchnormalization(size_t n_input,
                                Onnx__TensorProto **input,
                                size_t n_attribute,
                                Onnx__AttributeProto **attribute,
                                size_t n_output,
                                Onnx__TensorProto **output);

int operator_cast(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_conv(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_leakyrelu(size_t n_input,
                       Onnx__TensorProto **input,
                       size_t n_attribute,
                       Onnx__AttributeProto **attribute,
                       size_t n_output,
                       Onnx__TensorProto **output);

int operator_matmul(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_maxpool(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_mul(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output);

int operator_relu(size_t n_input,
                  Onnx__TensorProto **input,
                  size_t n_attribute,
                  Onnx__AttributeProto **attribute,
                  size_t n_output,
                  Onnx__TensorProto **output);

int operator_reshape(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_sigmoid(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_softmax(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_zipmap(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_sigmoid(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_softmax(size_t n_input,
                     Onnx__TensorProto **input,
                     size_t n_attribute,
                     Onnx__AttributeProto **attribute,
                     size_t n_output,
                     Onnx__TensorProto **output);

int operator_zipmap(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output);

int operator_quantizelinear(size_t n_input,
                            Onnx__TensorProto **input,
                            size_t n_attribute,
                            Onnx__AttributeProto **attribute,
                            size_t n_output,
                            Onnx__TensorProto **output);

int operator_convinteger(size_t n_input,
                         Onnx__TensorProto **input,
                         size_t n_attribute,
                         Onnx__AttributeProto **attribute,
                         size_t n_output,
                         Onnx__TensorProto **output);

int operator_matmulinteger(size_t n_input,
                           Onnx__TensorProto **input,
                           size_t n_attribute,
                           Onnx__AttributeProto **attribute,
                           size_t n_output,
                           Onnx__TensorProto **output);


#endif

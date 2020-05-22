#ifndef OPERATORS_H
#define OPERATORS_H
#include "onnx.pb-c.h"


typedef struct node_context  node_context;

struct node_context{
  Onnx__NodeProto     *onnx_node;
  Onnx__TensorProto  **inputs;
  Onnx__TensorProto  **outputs;
  int (*resolved_op)(node_context *ctx);
};

int operator_add(node_context *ctx);

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

int operator_conv(node_context *ctx);

int operator_leakyrelu(size_t n_input,
                       Onnx__TensorProto **input,
                       size_t n_attribute,
                       Onnx__AttributeProto **attribute,
                       size_t n_output,
                       Onnx__TensorProto **output);

int operator_matmul(node_context *ctx);

int operator_maxpool(node_context *ctx);

int operator_mul(size_t n_input,
                 Onnx__TensorProto **input,
                 size_t n_attribute,
                 Onnx__AttributeProto **attribute,
                 size_t n_output,
                 Onnx__TensorProto **output);

int operator_relu(node_context *ctx);

int operator_reshape(node_context *ctx);

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

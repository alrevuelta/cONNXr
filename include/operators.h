#ifndef OPERATORS_H
#define OPERATORS_H
#include "onnx.pb-c.h"

typedef struct node_context  node_context;

// TODO Move this to a file named operator_interface
struct node_context{
  Onnx__NodeProto     *onnx_node;
  Onnx__TensorProto  **inputs;
  Onnx__TensorProto  **outputs;
  int (*resolved_op)(node_context *ctx);
};

int operator_add(node_context *ctx);

int operator_argmax(node_context *ctx);

int operator_batchnormalization(node_context *ctx);

int operator_cast(node_context *ctx);

int operator_conv(node_context *ctx);

int operator_leakyrelu(node_context *ctx);

int operator_matmul(node_context *ctx);

int operator_maxpool(node_context *ctx);

int operator_mul(node_context *ctx);

int operator_relu(node_context *ctx);

int operator_reshape(node_context *ctx);

int operator_sigmoid(node_context *ctx);

int operator_softmax(node_context *ctx);

int operator_zipmap(node_context *ctx);

int operator_sigmoid(node_context *ctx);

int operator_softmax(node_context *ctx);

int operator_zipmap(node_context *ctx);

int operator_quantizelinear(node_context *ctx);

int operator_convinteger(node_context *ctx);

int operator_matmulinteger(node_context *ctx);


#endif

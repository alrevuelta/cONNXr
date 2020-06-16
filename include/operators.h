#ifndef OPERATORS_H
#define OPERATORS_H
#include "onnx.pb-c.h"
#include "operators/operator.h"

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

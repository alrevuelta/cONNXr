#ifndef INFERENCE_H
#define INFERENCE_H
#include "onnx.pb-c.h"
#include "operators/operator.h"

Onnx__TensorProto** inference(Onnx__ModelProto *model,
                              Onnx__TensorProto **inputs,
                              int nInputs);

void resolve(Onnx__ModelProto *model,
             Onnx__TensorProto **inputs,
             int nInputs);

/* TODO Shouldn't be global */
extern node_context all_context[50];
extern int _populatedIdx;

#endif

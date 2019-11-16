#ifndef EMBEDDEDML_INFERENCE_H
#define EMBEDDEDML_INFERENCE_H
#include "onnx.pb-c.h"

// TODO Hardcoded for initial tests
#define MAX_NUM_OF_OUTPUTS 20
Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS];
static int _outputIdx = 0;

// Investigate what to do with the output. Is it always a set of TensorProto?
Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs);

#endif

#ifndef EMBEDDEDML_INFERENCE_H
#define EMBEDDEDML_INFERENCE_H
#include "onnx.pb-c.h"

// Investigate what to do with the output. Is it always a set of TensorProto?
int inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs);

#endif

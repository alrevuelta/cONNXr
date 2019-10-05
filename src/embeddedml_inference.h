#ifndef EMBEDDEDML_INFERENCE_H
#define EMBEDDEDML_INFERENCE_H
#include "onnx.pb-c.h"

int inferenceFloat(float *input, int inputDim, Onnx__ModelProto *model);

#endif

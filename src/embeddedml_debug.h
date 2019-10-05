#ifndef EMBEDDEDML_DEBUG_H
#define EMBEDDEDML_DEBUG_H
#include "onnx.pb-c.h"

void Debug_PrintArray(float *array, int m, int n);
void Debug_PrintModelInformation(Onnx__ModelProto *model);

#endif

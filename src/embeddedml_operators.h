#ifndef EMBEDDEDML_OPERATORS_H
#define EMBEDDEDML_OPERATORS_H

#include "onnx.pb-c.h"

void Operators_MatMul_float(const float *in, const float *matrix, int m, int n, int k, float *out);
void Operators_MatMul_int(const int *in, const int *matrix, int m, int n, int k, int *out);

void Operators_Add_float(float *inOut, float *matrix, int m);
void Operators_Add_int(int *inOut, int *matrix, int m);



#endif

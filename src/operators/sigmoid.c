#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "sigmoid.h"

/*
void Operators_Sigmoid(void *x, int size)
{
  float *xf = (float*)x;
  while (size > 0) {
    size--;
    xf[size] = (1/(1 + exp(-(xf[size]))));
  }
}*/

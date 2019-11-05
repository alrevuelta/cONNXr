#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "argmax.h"

/* TODO
void Operators_ArgMax(void *x, int dimx, int dimy, int axis, int keepdims, int* out)
{
  // TODO keepdims is not used
  // TODO Only axis=1 is supported
  // Only 2D are supported
  float *xf = (float*)x;
  for (int i = 0; i < dimx; i++) {
    int argmaxindex = 0;
    float maxvalue = xf[i*dimy];
    for (int j = 0; j < dimy; j++) {
      if (xf[j+i*dimy] > maxvalue) {
        maxvalue = xf[j+i*dimy];
        argmaxindex = j;
      }
    }
    out[i] = argmaxindex;
  }
}*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_operators.h"

/*
For the first release only float will be supported. This means that other
types can work casting, but the performance will be bad, i.e. if int is used
it will be casted to float, and all the operations will be done in float type.
Once all operators in float are implemented, next step would be to support
other types.
*/

void Operators_MatMul_float(const float *a, const float *b, int m, int n, int k, float *c)
{
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0;
      for (int p = 0; p < n; p++) {
        sum += (a[i*n+p] * b[p*k+j]);
      }
      c[i*k+j] = sum;
    }
  }
}

void Operators_MatMul_int(const int *a, const int *b, int m, int n, int k, int *c)
{
}

void Operators_Add_float(float *inOut, float *matrix, int m)
{
  for (int i = 0; i < m; i++)
  {
    inOut[i] += matrix[i];
  }
}

void Operators_Add_int(int *inOut, int *matrix, int m)
{
  for (int i = 0; i < m; i++)
  {
    inOut[i] += matrix[i];
  }
}

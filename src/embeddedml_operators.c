#include "embeddedml_operators.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"

void Operators_MatMul_float(const float *in, const float *matrix, int m, int n, int k, float *out)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < k; j++)
    {
      out[i*m+j] = 0;
      for (int p = 0; p < n; p++)
      {
        out[i*m+j] += in[i*m+p] * matrix[p*k+j];
      }
    }
  }
}

void Operators_MatMul_int(const int *in, const int *matrix, int m, int n, int k, int *out)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < k; j++)
    {
      out[i*m+j] = 0;
      for (int p = 0; p < n; p++)
      {
        out[i*m+j] += in[i*m+p] * matrix[p*k+j];
      }
    }
  }
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

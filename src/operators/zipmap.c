#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../embeddedml_debug.h"
#include "operators.h"

int operator_zipmap(size_t n_input,
                    Onnx__TensorProto **input,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute,
                    size_t n_output,
                    Onnx__TensorProto **output)
{
  printf("Operator zipmap not implemented\n");
  return 1;
}

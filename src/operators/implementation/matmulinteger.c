#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "operators.h"

int operator_matmulinteger(size_t n_input,
                           Onnx__TensorProto **input,
                           size_t n_attribute,
                           Onnx__AttributeProto **attribute,
                           size_t n_output,
                           Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_matmulinteger\n");
  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  return 0;
}

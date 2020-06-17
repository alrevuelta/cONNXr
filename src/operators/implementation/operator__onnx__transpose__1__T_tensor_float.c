#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

operator_status operator__onnx__transpose__1__T_tensor_float(
    node_context *ctx
)
{
  TRACE_LEVEL0("Calling operator_transpose\n");

  //Onnx__TensorProto *A = searchInputByName(ctx, 0);
  //Onnx__TensorProto *B = searchInputByName(ctx, 1);

  //Onnx__TensorProto *Y = searchOutputByName(ctx, 0);

  return 0;
}

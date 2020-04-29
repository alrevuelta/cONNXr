#include <stdio.h>

#include "operators/operator_stub.h"

operator_status operator_stub(
    size_t n_input,
    Onnx__TensorProto** input,
    size_t n_attribute,
    Onnx__AttributeProto** attribute,
    size_t n_output,
    Onnx__TensorProto** output
) {
  fprintf(stderr, "operator not implemented!\n");
  exit(1);
}
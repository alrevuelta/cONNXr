#include <stdio.h>

#include "onnx.pb-c.h"

int operator_stub(
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
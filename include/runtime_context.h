#ifndef RUNTIME_CONTEXT_H
#define RUNTIME_CONTEXT_H

#include "onnx.pb-c.h"
#include "operators.h"
#include "operators/operator.h"

/* Quick test without pointers */

typedef struct runtime_context runtime_context;

struct runtime_context
{
    size_t            length;
    operator_context  contexts[50]; // TODO
};

runtime_context resolve_runtime_context(
  Onnx__ModelProto *model,
  Onnx__TensorProto **inputs,
  int n_inputs
);

#endif

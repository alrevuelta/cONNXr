#ifndef INFERENCE_H
#define INFERENCE_H
#include "pb/onnx.pb-c.h"
#include "operators/operators.h"

struct named_tensor {
  char                name[50];   // name given by model
  Onnx__TensorProto **tensor;     // points to the tensor in the context
};

struct output_tensors {
  struct named_tensor  named_tensors[30]; //todo quick test
  int                  n_tensors;         // number of tensors in the struct
};

extern struct output_tensors* tensor_table;

// Investigate what to do with the output. Is it always a set of TensorProto?
void inference(struct operator__context** all_op_context, int n_nodes);

struct operator__context** resolve_check_get_input_and_attr();

#endif

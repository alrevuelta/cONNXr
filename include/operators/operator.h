#ifndef OPERATOR_H
#define OPERATOR_H

#include "onnx.pb-c.h"
#include <stdbool.h>
#include <string.h>

typedef int (*onnx_operator)(
    size_t n_input,
    Onnx__TensorProto** input,
    size_t n_attribute,
    Onnx__AttributeProto** attribute,
    size_t n_output,
    Onnx__TensorProto** output
);

static
inline __attribute__((always_inline))
size_t operator_findTensors(
  Onnx__TensorProto    ** result,
  char                 ** names,
  size_t                  n_names,
  Onnx__TensorProto    ** tensors,
  size_t                  n_tensors
) {
  size_t n_results = 0;
  for (size_t i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
    for (size_t i_name = 0; i_name < n_names, i_name++) {
      if (strcmp(tensors[i_tensor]->name,names[i_name]) == 0) {
        result[n_results++] = tensors[i_tensor];
        if (n_results >= n_names)
          return n_results;
        break;
      }
    }
  }
  return n_results;
}

static
inline __attribute__((always_inline))
bool operator_tensorsAreOfSameType(
  Onnx__TensorProto    ** tensors,
  size_t                  n_tensors
) {
  if ( n_tensors < 2 ) return true;
  for (size_t i = 1; i < n_tensors; i++) {
    if ( tensors[0]->data_type != tensors[i]->data_type ) {
      return false;
    }
  }
  return true;
}

static
inline __attribute__((always_inline))
bool operator_tensorIsOneOfTypes(
  Onnx__TensorProto    * tensor,
  uint32_t             * types,
  size_t                 n_types
) {
  for (size_t i = 0; i < n_types; i++) {
    if ( tensor->data_type == types[i] ) {
      return true;
    }
  }
  return false;
}

#endif

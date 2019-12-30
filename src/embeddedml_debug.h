#ifndef EMBEDDEDML_DEBUG_H
#define EMBEDDEDML_DEBUG_H
#include "pb/onnx.pb-c.h"

#ifdef DEBUG
#define DEBUG_PRINT(FMT, ARGS...) do { \
  printf("%s:%d " FMT "\n", __FILE__, __LINE__, ## ARGS); \
  } while (0)
#else
  #define DEBUG_PRINT(fmt, ...){}
#endif

void debug_print_attributes(size_t n_attribute, Onnx__AttributeProto **attribute);
void debug_print_dims(size_t n_dims, int64_t *dims);
void debug_prettyprint_tensorproto(Onnx__TensorProto *tp);
void Debug_PrintArray(float *array, int m, int n);
void Debug_PrintModelInformation(Onnx__ModelProto *model);
void Debug_PrintTensorProto(Onnx__TensorProto *tp);
void debug_prettyprint_model(Onnx__ModelProto *model);

#endif

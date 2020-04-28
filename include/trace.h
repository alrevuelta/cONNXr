#ifndef TRACE_H
#define TRACE_H
#include "onnx.pb-c.h"

/* How to trace?
 * Use different macros for different tracing levels 0-2. When compiling, set
 * the corresponding flag to 1.
 * - LEVEL0: Easy to understand traces, without many detail. Operators that
 *           are called, dimensions,...
 * - LEVEL1: More detailed traces.
 * - LEVEL2: Very detailed traces.

 * If you want to always trace something, just use printf.
*/

#if TRACE_LEVEL>=0
#define TRACE_LEVEL0(FMT, ARGS...) do { \
  printf("[LEVEL0] %s:%d " FMT "", __FILE__, __LINE__, ## ARGS); \
  } while (0)
#else
  #define TRACE_LEVEL0(fmt, ...){}
#endif

#if TRACE_LEVEL>=1
#define TRACE_LEVEL1(FMT, ARGS...) do { \
  printf("[LEVEL1] %s:%d " FMT "", __FILE__, __LINE__, ## ARGS); \
  } while (0)
#else
  #define TRACE_LEVEL1(fmt, ...){}
#endif

#if TRACE_LEVEL>=2
#define TRACE_LEVEL2(FMT, ARGS...) do { \
  printf("[LEVEL2] %s:%d " FMT "", __FILE__, __LINE__, ## ARGS); \
  } while (0)
#else
  #define TRACE_LEVEL2(fmt, ...){}
#endif

void debug_print_attributes(size_t n_attribute, Onnx__AttributeProto **attribute);
void debug_print_dims(size_t n_dims, int64_t *dims);
void debug_prettyprint_tensorproto(Onnx__TensorProto *tp);
void Debug_PrintArray(float *array, int m, int n);
void Debug_PrintModelInformation(Onnx__ModelProto *model);
void Debug_PrintTensorProto(Onnx__TensorProto *tp);
void debug_prettyprint_model(Onnx__ModelProto *model);

#endif

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "onnx.pb-c.h"

#define FLOAT_TOLERANCE 0.001f

#define ASSERT_TRUE(expr)          \
    if (!(expr)){                  \
      printf("Error in assert\n"); \
      return -1;                   \
    }

// Compare if equal with some tolarenace
int compareAlmostEqualTensorProto(Onnx__TensorProto *a, Onnx__TensorProto *b);

int test_operator(char *outputName);

/* For a given onnx model and a set of inputs and expected outputs, runs
inference on that model and checks that the outputs are correct. The model_id
is used for prints and debugging purposes. Its also used by the python script
to check the inference time that it took to run that model.
*/
double test_model(
  char *model_id,
  char *model_path,
  char *io_path,
  int n_inputs,
  int n_outputs
);

#endif

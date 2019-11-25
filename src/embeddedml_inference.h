#ifndef EMBEDDEDML_INFERENCE_H
#define EMBEDDEDML_INFERENCE_H
#include "pb/onnx.pb-c.h"

// Just for some initial tests. Move to a common include
#include "operators/add.h"
#include "operators/argmax.h"
#include "operators/arrayfeatureextractor.h"
#include "operators/cast.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/maxpool.h"
#include "operators/relu.h"
#include "operators/reshape.h"
#include "operators/sigmoid.h"
#include "operators/softmax.h"
#include "operators/zipmap.h"

#include "operators/notimplemented.h"

// TODO Hardcoded for initial tests
#define MAX_NUM_OF_OUTPUTS 20
extern Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS];
extern int _outputIdx;

// Investigate what to do with the output. Is it always a set of TensorProto?
Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs);

typedef struct
{
  const char *name;
  void (*func)(size_t n_input,
               Onnx__TensorProto **input,
               size_t n_attribute,
               Onnx__AttributeProto **attribute,
               size_t n_output,
               Onnx__TensorProto **output);
} operatorptrs;

static const operatorptrs
          operatorsSet[] = {
                             {"Add", operator_add},
                             {"ArgMax", operator_argmax},
                             {"Cast", operator_cast},
                             {"Conv", operator_conv},
                             {"MatMul", operator_matmul},
                             {"MaxPool", operator_maxpool},
                             {"Relu", operator_relu},
                             {"Reshape", operator_reshape},

                            // If operator is not found notimplemented function is called
                           };

#endif

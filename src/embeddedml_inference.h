#ifndef EMBEDDEDML_INFERENCE_H
#define EMBEDDEDML_INFERENCE_H
#include "pb/onnx.pb-c.h"
#include "operators/operators.h"

// TODO Hardcoded for initial tests
#define MAX_NUM_OF_OUTPUTS 40
extern Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS];
extern int _outputIdx;

// Investigate what to do with the output. Is it always a set of TensorProto?
Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs);

typedef struct
{
  char *name;
  int (*func)( size_t n_input,
                Onnx__TensorProto **input,
                size_t n_attribute,
                Onnx__AttributeProto **attribute,
                size_t n_output,
               Onnx__TensorProto **output);
} operatorptrs;

static  operatorptrs
          operatorsSet[] = {
                             {"Add", operator_add},
                             {"ArgMax", operator_argmax},
                             {"Cast", operator_cast},
                             {"Conv", operator_conv},
                             {"MatMul", operator_matmul},
                             {"MaxPool", operator_maxpool},
                             {"Relu", operator_relu},
                             {"Reshape", operator_reshape},
                             {"BatchNormalization", operator_batchnormalization},
                             {"Mul", operator_mul},
                             {"LeakyRelu", operator_leakyrelu}
                           };

#endif

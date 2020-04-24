#ifndef INFERENCE_H
#define INFERENCE_H
#include "pb/onnx.pb-c.h"
#include "operators/operators.h"

// TODO Hardcoded for initial tests
#define MAX_NUM_OF_OUTPUTS 40
#define NUMBER_OF_OPERATORS 14
extern Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS];
extern int _outputIdx;

// Temporal tables to store the mapping between the output and the name
#define MY_TABLE_SIZE 30
#define MAX_STRING_SIZE 50
extern char lazy_output_mapping_names[MY_TABLE_SIZE][MAX_STRING_SIZE];
extern Onnx__TensorProto** lazy_outputs_mapping_tensors[MY_TABLE_SIZE];

// Investigate what to do with the output. Is it always a set of TensorProto?
Onnx__TensorProto** inference(struct operator__context** all_op_context, int n_nodes);

struct operator__context** resolve_check_get_input_and_attr();

#endif

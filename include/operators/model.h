#ifndef MODEL_H
#define MODEL_H

typedef struct model_input   model_input;
typedef struct model_output  model_output;
typedef struct model_context model_context;

#include "list.h"
#include "onnx.pb-c.h"
#include "operators/operator.h"

extern struct list_head model_context_list;
extern struct list_head model_input_list;
extern struct list_head model_output_list;

struct model_input {
    struct list_head list;
    Onnx__TensorProto *tensor;
};

struct model_output {
    struct list_head list;
    Onnx__TensorProto *tensor;
};

struct model_context {
    struct list_head list;
    operator_context *context;
};

model_input*
model_create_input(Onnx__TensorProto *input);

model_output*
model_create_output(Onnx__TensorProto *output);

model_context*
model_create_context(operator_context *ctx);

Onnx__TensorProto*
model_find_global_input(char *name);

Onnx__TensorProto*
model_find_global_output(char *name);

Onnx__TensorProto*
model_find_local_input(char *name);

Onnx__TensorProto*
model_find_local_output(char *name);


#endif
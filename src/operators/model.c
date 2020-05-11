#include "operators/model.h"
#include <stdlib.h>
#include <string.h>

struct list_head model_context_list;
struct list_head model_input_list;
struct list_head model_output_list;

model_input*
model_create_input(Onnx__TensorProto *tensor)
{
    model_input* input = malloc(sizeof(model_input));
    if(!input) {
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, strerror(errno));
        return NULL;
    }
    input->tensor = tensor;
    INIT_LIST_HEAD(&input->list);
    list_add(&input->list,&model_input_list);
    return input;
}

model_output*
model_create_output(Onnx__TensorProto *tensor)
{
    model_output* output = malloc(sizeof(model_output));
    if(!output) {
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, strerror(errno));
        return NULL;
    }
    output->tensor = tensor;
    INIT_LIST_HEAD(&output->list);
    list_add(&output->list,&model_output_list);
    return output;
}

model_context*
model_create_context(operator_context *ctx)
{
    model_context* context = malloc(sizeof(model_context));
    if(!context) {
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, strerror(errno));
        return NULL;
    }
    context->context = ctx;
    INIT_LIST_HEAD(&context->list);
    list_add(&context->list,&model_context_list);
    return context;
}

Onnx__TensorProto*
model_find_global_input(char *name) {
    model_input *input;
    list_for_each_entry(input, &model_input_list, list)
    {
        if (strcmp(name, input->tensor->name)==0) {
            return input->tensor;
        }
    }
    return NULL;
}

Onnx__TensorProto*
model_find_global_output(char *name) {
    model_output *output;
    list_for_each_entry(output, &model_output_list, list)
    {
        if (strcmp(name, output->tensor->name)==0) {
            return output->tensor;
        }
    }
    return NULL;
}

Onnx__TensorProto*
model_find_local_input(char *name) {
    model_context *pos;
    list_for_each_entry(pos, &model_context_list, list)
    {
        for (size_t i_tensor = 0; i_tensor < pos->context->node->n_input; i_tensor++)
        {
            if (strcmp(name, pos->context->input[i_tensor]->name)==0) {
                return pos->context->input[i_tensor];
            }
        }
    }
    return NULL;
}

Onnx__TensorProto*
model_find_local_output(char *name) {
    model_context *pos;
    list_for_each_entry(pos, &model_context_list, list)
    {
        for (size_t i_tensor = 0; i_tensor < pos->context->node->n_output; i_tensor++)
        {
            if (strcmp(name, pos->context->output[i_tensor]->name)==0) {
                return pos->context->output[i_tensor];
            }
        }
    }
    return NULL;
}
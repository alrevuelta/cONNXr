#include "operators/operator.h"
#include "operators/operator_info.h"
#include "operators/operator_stub.h"
#include "operators/model.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

operator_context*
operator_context_create(operator_info    *info,
                        Onnx__NodeProto  *node)
{
    size_t n_input     = max(info->n_input,node->n_input);
    size_t n_output    = max(info->n_output,node->n_output);
    size_t n_attribute = max(info->n_attribute,node->n_attribute);
    operator_context      *ctx       = calloc(1,
                                              sizeof(operator_context));
    Onnx__TensorProto    **input     = calloc(n_input,
                                              sizeof(Onnx__TensorProto*));
    Onnx__TensorProto    **output    = calloc(n_output,
                                              sizeof(Onnx__TensorProto*));
    Onnx__AttributeProto **attribute = calloc(n_attribute,
                                              sizeof(Onnx__AttributeProto*));

    if (!ctx || !input || !output || !attribute) {
        fprintf(stderr,
                "%s:%s out of mem, could not create context\n",
                node->op_type,
                node->name);
        if (ctx) {
            free(ctx);
        }
        if (input) {
            free(input);
        }
        if (output) {
            free(output);
        }
        if (attribute) {
            free(attribute);
        }
        return NULL;
    }
    /* create outputs */
    for (size_t i_output = 0; i_output < node->n_output; i_output++) {
        if (strcmp(node->output[i_output],"")==0) {
            output[i_output] = NULL;
            continue;
        }
        output[i_output] = calloc(1,sizeof(Onnx__TensorProto*));
        if (!output[i_output]) {
            fprintf(stderr,
                    "%s:%s out of mem, could not create context\n",
                    node->op_type,
                    node->name);
            for (;i_output;i_output--) {
                if (output[i_output]) {
                    free(output[i_output]);
                }
            }
            return NULL;
        }
        output[i_output]->name = node->output[i_output];
    }

    /* create attributes */
    for (size_t i_iattr = 0; i_iattr < info->n_attribute; i_iattr++) {
        operator_info_attribute *iattr = &info->attribute[i_iattr];
        attribute[i_iattr] = NULL;
        for (size_t i_nattr = 0; i_nattr < node->n_attribute; i_nattr++) {
            Onnx__AttributeProto *nattr = node->attribute[i_nattr];
            if (strcmp(iattr->name,nattr->name)==0) {
                attribute[i_iattr] = node->attribute[i_nattr];
                break;
            }
        }
    }

    ctx->input     = input;
    ctx->output    = output;
    ctx->attribute = attribute;
    ctx->node      = node;
    ctx->info      = info;
    ctx->executor  = NULL;

    model_create_context(ctx);
    return ctx;
}

bool
operator_context_link(Onnx__ModelProto *model,
                      operator_context *ctx)
{
    bool valid = true;
    for(size_t i_tensor = 0; i_tensor < ctx->node->n_input; i_tensor++) {
        char *name = ctx->node->input[i_tensor];
        Onnx__TensorProto *tensor = NULL;

        tensor = model_find_global_input(name);
        if (tensor) {
            ctx->input[i_tensor] = tensor;
            continue;
        }
        tensor = model_find_local_output(name);
        if (tensor) {
            ctx->input[i_tensor] = tensor;
            continue;
        }
        valid = false;
        fprintf(stderr,
                "could not find tensor '%s' for node '%s:%s'\n",
                name,
                ctx->node->op_type,
                ctx->node->name);
    }
    return valid;
}

bool
operator_context_linkAll(Onnx__ModelProto *model)
{
    bool valid = true;
    model_context *pos;
    list_for_each_entry(pos,&model_context_list,list) {
        valid &= operator_context_link(model, pos->context);
    }
    return valid;
}

bool
operator_context_createAll(Onnx__ModelProto *model)
{
    bool valid = true;
    for (size_t i_node = 0; i_node < model->graph->n_node; i_node++) {
        Onnx__NodeProto *node = model->graph->node[i_node];
        operator_info   *info = NULL;
        for(size_t i_opset = 0; i_opset < model->n_opset_import; i_opset++) {
            if (strcmp(node->domain,model->opset_import[i_opset]->domain)!=0) {
                continue;
            }
            info = operator_info_find(model->opset_import[i_opset]->version,
                                      node->domain,
                                      node->op_type);
            if (info) {
                break;
            }
        }
        if (!info) {
            fprintf(stderr,
                    "%s:%s operator not found! will be stubbed\n",
                    node->op_type,
                    node->name);
            info  = &operator_stub_info;
            valid = false;
        }
        operator_context *ctx = operator_context_create(info, node);
        if (!ctx) {
            return false;
        }
    }
    return valid;
}
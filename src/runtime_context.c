#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime_context.h"
#include "onnx.pb-c.h"
#include "operators.h"
#include "utils.h"
#include "trace.h"
#include "operators/operator.h"

#include "operators/onnx/operator__onnx__add__7.h"
#include "operators/onnx/operator__onnx__relu__6.h"
#include "operators/onnx/operator__onnx__conv__11.h"
#include "operators/onnx/operator__onnx__matmul__9.h"
#include "operators/onnx/operator__onnx__reshape__5.h"
#include "operators/onnx/operator__onnx__maxpool__11.h"
#include "operators/operator_stub.h"


# define POPULATE_INPUT(TENSOR, INDEX){ \
  /* Search and store the input for a given operator */ \
  local_ctx.input->TENSOR = malloc(sizeof(*local_ctx.input->TENSOR)); \
  local_ctx.input->TENSOR->tensor = malloc(sizeof(*local_ctx.input->TENSOR->tensor)); \
  local_ctx.input->TENSOR->tensor = NULL; \
  /* Input size is variable, check that the index is contained */ \
  if (model->graph->node[nodeIdx]->n_input >= (INDEX + 1)){ \
    local_ctx.input->TENSOR->tensor = searchTensorProtoByName(model, \
                                   inputs, \
                                   n_inputs, \
                                   model->graph->node[nodeIdx]->input[INDEX], \
                                   runtime_outputs); \
  } \
}

# define POPULATE_ATTRIBUTE(ATTRIBUTE) { \
  printf("Populating attributes %s\n", #ATTRIBUTE); \
  local_ctx.attribute->auto_pad = searchAttributeNyName( \
              model->graph->node[nodeIdx]->n_attribute, \
              model->graph->node[nodeIdx]->attribute, #ATTRIBUTE); \
}

#define STORE_OUTPUT(TENSOR, INDEX) {               \
  /* TODO: Index is not used. Only 1 output is assumed */ \
  /* Set the name of the output */ \
  printf("Storing output %s in table at position %zu\n", \
          model->graph->node[nodeIdx]->output[0], \
          runtime_outputs->length); \
  local_ctx.output->TENSOR = malloc(sizeof(*local_ctx.output->TENSOR));                     \
  local_ctx.output->TENSOR->tensor = malloc(sizeof(*local_ctx.output->TENSOR->tensor));     \
  local_ctx.output->TENSOR->tensor->name = malloc(sizeof(char) * 60);                  \
  strcpy(local_ctx.output->TENSOR->tensor->name,                                       \
         model->graph->node[nodeIdx]->output[0]);                                 \
  printf("Stored ok %s\n", local_ctx.output->TENSOR->tensor->name);                    \
  /* Store the output name and reference in a table */ \
  size_t tensor_idx = runtime_outputs->length; \
  runtime_outputs->tensors[tensor_idx].tensor = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].tensor)); \
  runtime_outputs->tensors[tensor_idx].tensor = local_ctx.output->TENSOR->tensor; \
  runtime_outputs->tensors[tensor_idx].name = malloc(sizeof(char) * 60); \
  strcpy(runtime_outputs->tensors[tensor_idx].name, \
         model->graph->node[nodeIdx]->output[0]); \
  runtime_outputs->length++; \
}

#define ALLOC_MEM(){ \
  local_ctx.input = malloc(sizeof(*local_ctx.input)); \
  local_ctx.output = malloc(sizeof(*local_ctx.output)); \
  local_ctx.attribute = malloc(sizeof(*local_ctx.attribute)); \
}

runtime_context resolve_runtime_context(
  Onnx__ModelProto *model,
  Onnx__TensorProto **inputs,
  int n_inputs,
  runtime_outputs *runtime_outputs
) {

  printf("\n\n\nResolving runtime context\n");

  /* Not mallocing to simplify for an initial test */
  runtime_context rc;

  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    printf("Operator at %d is %s\n", nodeIdx, model->graph->node[nodeIdx]->op_type);

    /* New idea prototyping, just a proof of concept */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      printf("Operator at %d is Add\n", nodeIdx);
      operator_context__operator__onnx__add__7 local_ctx;

      ALLOC_MEM();
      POPULATE_INPUT(A, 0);
      POPULATE_INPUT(B, 1);
      STORE_OUTPUT(C, 0);

      local_ctx.operator = resolve_operator__onnx__add__7(&local_ctx);

      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      printf("Operator at %d is Conv\n", nodeIdx);
      operator_context__operator__onnx__conv__11 local_ctx;

      ALLOC_MEM();

      POPULATE_INPUT(X, 0);
      POPULATE_INPUT(W, 1);
      POPULATE_INPUT(B, 2);

      POPULATE_ATTRIBUTE(auto_pad);
      POPULATE_ATTRIBUTE(dilations);
      POPULATE_ATTRIBUTE(group);
      POPULATE_ATTRIBUTE(kernel_shape);
      POPULATE_ATTRIBUTE(pads);
      POPULATE_ATTRIBUTE(strides);

      STORE_OUTPUT(Y, 0);

      local_ctx.operator = resolve_operator__onnx__conv__11(&local_ctx);
      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      printf("Operator at %d is Reshape\n", nodeIdx);
      operator_context__operator__onnx__reshape__5 local_ctx;

      ALLOC_MEM();
      POPULATE_INPUT(data, 0);
      POPULATE_INPUT(shape, 1);
      STORE_OUTPUT(reshaped, 0);

      local_ctx.operator = resolve_operator__onnx__reshape__5(&local_ctx);
      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      printf("Operator at %d is Relu\n", nodeIdx);
      operator_context__operator__onnx__relu__6 local_ctx;

      ALLOC_MEM();
      POPULATE_INPUT(X, 0);
      STORE_OUTPUT(Y, 0);

      local_ctx.operator = resolve_operator__onnx__relu__6(&local_ctx);
      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MaxPool")){
      printf("Operator at %d is Maxpool\n", nodeIdx);
      operator_context__operator__onnx__maxpool__11 local_ctx;

      ALLOC_MEM();
      POPULATE_INPUT(X, 0);
      POPULATE_ATTRIBUTE(auto_pad);
      POPULATE_ATTRIBUTE(dilations);
      POPULATE_ATTRIBUTE(kernel_shape);
      POPULATE_ATTRIBUTE(pads);
      POPULATE_ATTRIBUTE(strides);
      STORE_OUTPUT(Y, 0);

      local_ctx.operator = resolve_operator__onnx__maxpool__11(&local_ctx);
      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MatMul")){


      printf("debug %s\n", runtime_outputs->tensors[9].tensor->name);

      printf("Operator at %d is Matmul\n", nodeIdx);
      operator_context__operator__onnx__matmul__9 local_ctx;

      ALLOC_MEM();
      POPULATE_INPUT(A, 0);
      POPULATE_INPUT(B, 1);
      STORE_OUTPUT(Y, 0);

      printf("debug2 %s\n", runtime_outputs->tensors[9].tensor->name);

      local_ctx.operator = resolve_operator__onnx__matmul__9(&local_ctx);
      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);
   }
  }


  return rc;
}

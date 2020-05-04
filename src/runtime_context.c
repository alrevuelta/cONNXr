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

      /* Cast the context into the specific one */
      operator_context__operator__onnx__add__7 local_ctx;

      local_ctx.input = malloc(sizeof(*local_ctx.input));
      local_ctx.input->A = malloc(sizeof(*local_ctx.input->A));
      local_ctx.input->B = malloc(sizeof(*local_ctx.input->B));
      local_ctx.input->A->tensor = malloc(sizeof(*local_ctx.input->A->tensor));
      local_ctx.input->B->tensor = malloc(sizeof(*local_ctx.input->B->tensor));

      /* Resolve the inputs */
      local_ctx.input->A->tensor = searchTensorProtoByName(model,
                                     inputs,
                                     n_inputs,
                                     model->graph->node[nodeIdx]->input[0],
                                     runtime_outputs);

      local_ctx.input->B->tensor = searchTensorProtoByName(model,
                                    inputs,
                                    n_inputs,
                                    model->graph->node[nodeIdx]->input[1],
                                    runtime_outputs);

      /* Resolve the attributes */
      /* None */

      /* Set output name */
      local_ctx.output->C = malloc(sizeof(*local_ctx.output->C));
      local_ctx.output->C->tensor = malloc(sizeof(*local_ctx.output->C->tensor));
      local_ctx.output->C->tensor->name = malloc(sizeof(char) * 40);
      strcpy(local_ctx.output->C->tensor->name,
             model->graph->node[nodeIdx]->output[0]);

      /* Resolve the operator */
      local_ctx.operator = resolve_operator__onnx__add__7(&local_ctx);

      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

      size_t tensor_idx = runtime_outputs->length;
      runtime_outputs->tensors[tensor_idx].tensor = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].tensor));
      runtime_outputs->tensors[tensor_idx].tensor = local_ctx.output->C->tensor;
      runtime_outputs->tensors[tensor_idx].name = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].name));
      strcpy(runtime_outputs->tensors[tensor_idx].name,
             model->graph->node[nodeIdx]->output[0]);
      runtime_outputs->length++;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      printf("Operator at %d is Conv\n", nodeIdx);

      operator_context__operator__onnx__conv__11 local_ctx;

      local_ctx.input = malloc(sizeof(*local_ctx.input));
      local_ctx.input->X = malloc(sizeof(*local_ctx.input->X));
      local_ctx.input->X->tensor = malloc(sizeof(*local_ctx.input->X->tensor));

      local_ctx.input->W = malloc(sizeof(*local_ctx.input->W));
      local_ctx.input->W->tensor = malloc(sizeof(*local_ctx.input->W->tensor));

      local_ctx.input->B = malloc(sizeof(*local_ctx.input->B));
      local_ctx.input->B->tensor = malloc(sizeof(*local_ctx.input->B->tensor));

      local_ctx.input->X->tensor = NULL;
      local_ctx.input->W->tensor = NULL;
      local_ctx.input->B->tensor = NULL;

      /* Quick workaround to avoid accessing an element out of bound */
      if (model->graph->node[nodeIdx]->n_input >= 1){
        local_ctx.input->X->tensor = searchTensorProtoByName(model,
                                       inputs,
                                       n_inputs,
                                       model->graph->node[nodeIdx]->input[0],
                                       runtime_outputs);
      }
      if (model->graph->node[nodeIdx]->n_input >= 2){
        local_ctx.input->W->tensor = searchTensorProtoByName(model,
                                      inputs,
                                      n_inputs,
                                      model->graph->node[nodeIdx]->input[1],
                                      runtime_outputs);
      }
      if (model->graph->node[nodeIdx]->n_input >= 3){
        local_ctx.input->B->tensor = searchTensorProtoByName(model,
                                       inputs,
                                       n_inputs,
                                       model->graph->node[nodeIdx]->input[2],
                                       runtime_outputs);
      }

      // Attributes. NULL is returned if not found.
      local_ctx.attribute = malloc(sizeof(*local_ctx.attribute));

      local_ctx.attribute->auto_pad = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "auto_pad");
      local_ctx.attribute->dilations = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "dilations");
      local_ctx.attribute->group = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "group");
      local_ctx.attribute->kernel_shape = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "kernel_shape");
      local_ctx.attribute->pads = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "pads");
      local_ctx.attribute->strides = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "strides");

      /* Set output name */
      //local_ctx.output = malloc(sizeof(*local_ctx.output));
      local_ctx.output->Y = malloc(sizeof(*local_ctx.output->Y));
      local_ctx.output->Y->tensor = malloc(sizeof(*local_ctx.output->Y->tensor));
      local_ctx.output->Y->tensor->name = malloc(sizeof(char) * 40);
      strcpy(local_ctx.output->Y->tensor->name,
             model->graph->node[nodeIdx]->output[0]);

      local_ctx.operator = resolve_operator__onnx__conv__11(&local_ctx);

      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

      size_t tensor_idx = runtime_outputs->length;
      runtime_outputs->tensors[tensor_idx].tensor = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].tensor));
      runtime_outputs->tensors[tensor_idx].tensor = local_ctx.output->Y->tensor;
      runtime_outputs->tensors[tensor_idx].name = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].name));
      strcpy(runtime_outputs->tensors[tensor_idx].name,
             model->graph->node[nodeIdx]->output[0]);
      runtime_outputs->length++;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      printf("Operator at %d is Reshape\n", nodeIdx);

      operator_context__operator__onnx__reshape__5 local_ctx;

      local_ctx.input = malloc(sizeof(*local_ctx.input));

      local_ctx.input->data = malloc(sizeof(*local_ctx.input->data));
      local_ctx.input->data->tensor = malloc(sizeof(*local_ctx.input->data->tensor));
      local_ctx.input->data->tensor = searchTensorProtoByName(model,
                                     inputs,
                                     n_inputs,
                                     model->graph->node[nodeIdx]->input[0],
                                     runtime_outputs);

      local_ctx.input->shape = malloc(sizeof(*local_ctx.input->shape));
      local_ctx.input->shape->tensor = malloc(sizeof(*local_ctx.input->shape->tensor));
      local_ctx.input->shape->tensor = searchTensorProtoByName(model,
                                     inputs,
                                     n_inputs,
                                     model->graph->node[nodeIdx]->input[1],

                                     runtime_outputs);

      // no attr

      /* Set output name */
      local_ctx.output = malloc(sizeof(*local_ctx.output));
      local_ctx.output->reshaped = malloc(sizeof(*local_ctx.output->reshaped));
      local_ctx.output->reshaped->tensor = malloc(sizeof(*local_ctx.output->reshaped->tensor));
      local_ctx.output->reshaped->tensor->name = malloc(sizeof(char) * 40);
      strcpy(local_ctx.output->reshaped->tensor->name,
             model->graph->node[nodeIdx]->output[0]);


      local_ctx.operator = resolve_operator__onnx__reshape__5(&local_ctx);

      rc.contexts[nodeIdx] = *(operator_context*)(&local_ctx);

      size_t tensor_idx = runtime_outputs->length;
      runtime_outputs->tensors[tensor_idx].tensor = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].tensor));
      runtime_outputs->tensors[tensor_idx].tensor = local_ctx.output->reshaped->tensor;
      runtime_outputs->tensors[tensor_idx].name = malloc(sizeof(*runtime_outputs->tensors[tensor_idx].name));
      strcpy(runtime_outputs->tensors[tensor_idx].name,
             model->graph->node[nodeIdx]->output[0]);
      runtime_outputs->length++;
    }
  }


  return rc;
}

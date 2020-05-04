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
#include "operators/operator_stub.h"

runtime_context resolve_runtime_context(
  Onnx__ModelProto *model,
  Onnx__TensorProto **inputs,
  int n_inputs
) {

  printf("\n\n\n\nTODO\n");

  /* Not mallocing to simplify for an initial test */
  runtime_context rc;

  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    /* New idea prototyping, just a proof of concept */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      printf("Operator at %d is Add\n", nodeIdx);

      /* Cast the context into the specific one */
      operator_context__operator__onnx__add__7 local_ctx;

      /* Resolve the inputs */
      local_ctx.input->A->tensor = searchTensorProtoByName(model,
                                     inputs,
                                     n_inputs,
                                     model->graph->node[nodeIdx]->input[0]);

      local_ctx.input->B->tensor = searchTensorProtoByName(model,
                                    inputs,
                                    n_inputs,
                                    model->graph->node[nodeIdx]->input[1]);

      /* Resolve the attributes */
      /* None */

      /* Set output name */
      local_ctx.output->C->tensor->name = malloc(sizeof(char) * 40);
      strcpy(local_ctx.output->C->tensor->name,
             model->graph->node[nodeIdx]->output[0]);

      /* Resolve the operator */
      local_ctx.operator = resolve_operator__onnx__add__7(&local_ctx);

      /* Save the relation between output and name in a table */

      /* Old code
      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->C;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;
      all_op_context[nodeIdx] = (struct operator__context*)c;*/

      operator_context asd = (operator_context)local_ctx;
      //rc.contexts[nodeIdx] = &asd;

    }
    /* ... */
  }


  return rc;
}

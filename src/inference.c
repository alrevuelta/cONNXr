#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "inference.h"
#include "operators/operator_sets.h"

// Won't be global in the future
node_context all_context[50];
int _populatedIdx = 0;

void resolve(Onnx__ModelProto *model,
             Onnx__TensorProto **inputs,
             int nInputs)
{
  /* Resolving operators and input/outputs. Has to be moved outside of infeference */
  TRACE_LEVEL0("Resolving\n");
  _populatedIdx = -1;
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    all_context[nodeIdx].onnx_node = model->graph->node[nodeIdx];

    // Search the inputs for a node
    all_context[nodeIdx].inputs = malloc(sizeof(Onnx__TensorProto) * model->graph->node[nodeIdx]->n_input);
    for (int i = 0; i < model->graph->node[nodeIdx]->n_input; i++)
    {
      all_context[nodeIdx].inputs[i] = malloc(sizeof(Onnx__TensorProto));
      all_context[nodeIdx].inputs[i] = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[i]);
    }

    // Allocate memory for future outputs and set the name
    all_context[nodeIdx].outputs = malloc(sizeof(Onnx__TensorProto) * model->graph->node[nodeIdx]->n_output);
    for (int i = 0; i < model->graph->node[nodeIdx]->n_output; i++)
    {
      all_context[nodeIdx].outputs[i] = malloc(sizeof(Onnx__TensorProto));
      all_context[nodeIdx].outputs[i]->name = malloc(sizeof(char) * 50);
      strcpy(all_context[nodeIdx].outputs[i]->name, model->graph->node[nodeIdx]->output[i]);
    }

    /*** Prototyping ***/
    // Check model->opset_import->has_version must be True
    // More than 1 opset can be imported. Iterate n_opset_import
    // model->opset_import[0]->version
    // TODO Hackish temporal solution. Use opset 12.
    operator_resolver resolver = find_operator_resolver(model->graph->node[nodeIdx]->op_type, 12);
    operator_executer executer = resolver(&all_context[nodeIdx]);
    all_context[nodeIdx].resolved_op = executer;
    _populatedIdx++;
  }
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  TRACE_LEVEL0("\n\nCalling inference\n");
  TRACE_LEVEL0("The graph has nodes=%zu\n", model->graph->n_node);

  /* Run inference */
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    printf("inference on node \n");
    all_context[nodeIdx].resolved_op(&all_context[nodeIdx]);
  }

  // TODO
  return 0;
}

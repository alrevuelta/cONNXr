#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "tracing.h"
#include "inference.h"
#include "operators/operator_set.h"

// Won't be global in the future
node_context all_context[MAX_NUM_OF_NODES];
int _populatedIdx = 0;

void resolve(Onnx__ModelProto *model,
             Onnx__TensorProto **inputs,
             int nInputs)
{
  TRACE_ENTRY(1);
  /* Resolving operators and input/outputs. Has to be moved outside of infeference */
  _populatedIdx = -1;

  TRACE_FATAL(0, model->graph->n_node > MAX_NUM_OF_NODES, "The number of nodes of the model is greater than the hardcoded one");

  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    all_context[nodeIdx].onnx_node = model->graph->node[nodeIdx];

    // Search the inputs for a node
    all_context[nodeIdx].inputs = malloc(sizeof(Onnx__TensorProto) * model->graph->node[nodeIdx]->n_input);
    for (int i = 0; i < model->graph->node[nodeIdx]->n_input; i++)
    {
      all_context[nodeIdx].inputs[i] = malloc(sizeof(Onnx__TensorProto));
      all_context[nodeIdx].inputs[i] = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[i]);
      if (all_context[nodeIdx].inputs[i] && all_context[nodeIdx].inputs[i]->has_raw_data){
        /* If the tensor has raw data, deserialize it */
        TRACE(1, true, "input %s has raw data", all_context[nodeIdx].inputs[i]->name);
        // TODO: Not tested. Crashing but currently not needed
        convertRawDataOfTensorProto(all_context[nodeIdx].inputs[i]);
      }
    }

    // Allocate memory for future outputs and set the name
    all_context[nodeIdx].outputs = malloc(sizeof(Onnx__TensorProto) * model->graph->node[nodeIdx]->n_output);
    for (int i = 0; i < model->graph->node[nodeIdx]->n_output; i++)
    {
      all_context[nodeIdx].outputs[i] = malloc(sizeof(Onnx__TensorProto));
      init_tensor_proto(all_context[nodeIdx].outputs[i]);
      all_context[nodeIdx].outputs[i]->name = strdup(model->graph->node[nodeIdx]->output[i]);

      // TODO This is unset at this point but set afterward inside each
      // function. However there is a problem because some node output
      // is some node else input. Hence if the type is unset it can't
      // be resolved. Hardcoded to FLOAT but this is a HUGE TODO
      all_context[nodeIdx].outputs[i]->data_type = 1;
    }

    /*** Prototyping ***/
    // Check model->opset_import->has_version must be True
    // More than 1 opset can be imported. Iterate n_opset_import
    // model->opset_import[0]->version
    // TODO Hackish temporal solution. Use opset 12.
    size_t version = 12;
    operator_preparer prepare = operator_set_find_preparer(model->graph->node[nodeIdx]->op_type, version);
    TRACE_FATAL(0, !prepare, "No prepare function could be found for operator '%s' version '%zu'", model->graph->node[nodeIdx]->op_type, version);
    prepare(&all_context[nodeIdx]);
    _populatedIdx++;
  }
  TRACE_EXIT(1);
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  TRACE_ENTRY(1);
  TRACE(1, true, "The graph has nodes=%zu", model->graph->n_node);

  /* Run inference */
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    TRACE(1, true, "Running node %d, operator=%s", nodeIdx, model->graph->node[nodeIdx]->op_type);
    for (node_context *ctx = &all_context[nodeIdx]; ctx; ctx=ctx->next) {
      if (!ctx->threadsafe) {
        // wait for all threads to finish
      }
      // issue new thread
      ctx->executer(ctx);
    }
    // wait for all threads to finish
  }

  // TODO
  TRACE_EXIT(1);
  return 0;
}

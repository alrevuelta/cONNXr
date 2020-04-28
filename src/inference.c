#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "inference.h"

int _outputIdx = 0;
Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS] = {};

static int call_operator(char *name,
                         size_t n_input,
                         Onnx__TensorProto **input,
                         size_t n_attribute,
                         Onnx__AttributeProto **attribute,
                         size_t n_output,
                         Onnx__TensorProto **output)
{
  int i;
  for(i = 0; i < NUMBER_OF_OPERATORS; i++) {
    if(strcmp(operatorsSet[i].name, name) == 0) {
      int ret = operatorsSet[i].func(n_input,
                                     input,
                                     n_attribute,
                                     attribute,
                                     n_output,
                                     output);
      return ret;
    }
  }
  /* If it reaches this point, the operator wasnt found. Throw error? */
  TRACE_LEVEL0("\n\nTODO: Operator %s doest not exist or its not implemented\n\n", name);

  /* break */
  perror("There was an error calling the operator");
  exit(1);

  return -1;
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  //int error = 0;

  /* Dirty trick to allow multiple runs. There is a memory leak for sure */
  _outputIdx = 0;
  TRACE_LEVEL0("Calling inference\n");
  TRACE_LEVEL0("The graph has nodes=%zu\n", model->graph->n_node);

  // Iterate all nodes in the graph
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    char *operation = model->graph->node[nodeIdx]->op_type;
    TRACE_LEVEL0("node=%d, operation=%s, n_input=%zu, n_output=%zu\n",
                 nodeIdx,
                 model->graph->node[nodeIdx]->op_type,
                 model->graph->node[nodeIdx]->n_input,
                 model->graph->node[nodeIdx]->n_output);

    // TODO hardcoded to one output
    size_t nOutputs = 1;
    Onnx__TensorProto *out0 = malloc(sizeof(*out0));

    /* Do this alloc inside the operator ?*/
    Onnx__TensorProto **nodeOutputs = malloc(sizeof(*out0));

    nodeOutputs[0] = out0;
    Onnx__TensorProto **nodeInputs = malloc(sizeof(*out0) * model->graph->node[nodeIdx]->n_input);

    // Populate the input array by gathering all the required inputs
    for (int inp = 0; inp < model->graph->node[nodeIdx]->n_input; ++inp) {
      Onnx__TensorProto *inpN =malloc(sizeof(*inpN));
      inpN = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[inp]);
      nodeInputs[inp] = inpN;
      //printf("\n %d\n", model->graph->node[nodeIdx]->n_input);
    }

    int error = call_operator(operation,
                              model->graph->node[nodeIdx]->n_input,
                              nodeInputs,
                              model->graph->node[nodeIdx]->n_attribute,
                              model->graph->node[nodeIdx]->attribute,
                              nOutputs,// TODO use model->graph->node[nodeIdx]->n_output
                              nodeOutputs);

    if (error){
      perror("There was an error calling the operator");
      exit(1);
    }

    /* Reuse the string */
    out0->name = model->graph->node[nodeIdx]->output[0];
    TRACE_LEVEL0("Storing output in list index=%d, name=%s\n", _outputIdx, out0->name);

    /* TODO this is hardcoded */
    _outputs[_outputIdx++] = nodeOutputs[0];
  }

  // TODO:
  // Free calculaterTensors memory
  // Free also extra allocations within the structure (i.e. doubles...)

  return _outputs;
}

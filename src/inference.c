#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "inference.h"

int _outputIdx = 0;
Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS] = {};

node_context all_context[50];
int _populatedIdx = -1;

#if 0
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
#endif

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  /* Dirty trick to allow multiple runs. There is a memory leak for sure */
  _outputIdx = 0;


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

    // Resolve operator. Hardcoded for MNIST

    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      all_context[nodeIdx].resolved_op = &operator_add;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      all_context[nodeIdx].resolved_op = &operator_conv;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      all_context[nodeIdx].resolved_op = &operator_reshape;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      all_context[nodeIdx].resolved_op = &operator_relu;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MaxPool")){
      all_context[nodeIdx].resolved_op = &operator_maxpool;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MatMul")){
      all_context[nodeIdx].resolved_op = &operator_matmul;
   }
    _populatedIdx++;
  }

  TRACE_LEVEL0("\n\nCalling inference\n");
  TRACE_LEVEL0("The graph has nodes=%zu\n", model->graph->n_node);

  /* Run inference */
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    all_context[nodeIdx].resolved_op(&all_context[nodeIdx]);
  }


  // Print the output for MNIST
  printf("\n\nFinal output name = %s\n\n", all_context[11].outputs[0]->name);
  for (int i = 0; i < all_context[11].outputs[0]->n_float_data; i++){
    printf("n_float_data[%d] = %f\n", i, all_context[11].outputs[0]->float_data[i]);
  }


#if 0
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

  #endif

  // TODO:
  // Free calculaterTensors memory
  // Free also extra allocations within the structure (i.e. doubles...)

  return _outputs;
}

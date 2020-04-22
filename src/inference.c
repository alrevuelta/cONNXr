#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pb/onnx.pb-c.h"
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

  /* New approach, prototype */
  /* Not really going to be called here, but when loading the model. Placed
  here to simplify things.*/
  operator__context** all_op_context;
  all_op_context = resolve_check_get_input_and_attr(model,
                                                    inputs,
                                                    nInputs);

  // Iterate all nodes in the graph
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    /* The operators inputs havent been modified yet */
    /* See Add as reference */
    all_op_context.run(all_op_context);

    /* Legacy code */
    #if 0
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

    #endif
  }

  // TODO:
  // Free calculaterTensors memory
  // Free also extra allocations within the structure (i.e. doubles...)

  return _outputs;
}

/* Called once the model is loaded. Resolved each operator to a its function
and popualtes all structures with the inputs */
operator__context** resolve_check_get_input_and_attr(
                                     Onnx__ModelProto *model,
                                     Onnx__TensorProto **inputs,
                                     int nInputs)
{
  operator__context **all_op_context;
  /* malloc space for model->graph->n_node
  *  malloc also space for TensorProto or AttributeProto depending on the
  *  operator ?
  */
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    /* New idea prototyping, just a proof of concept */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      printf("Calling operator Add\n");
      operator__onnx__add__input *i;
      operator__onnx__add__output *o;
      operator__onnx__add__attribute *a;

      /* Find input 0 (which is A) and in mnist will have different
      names depending on the node:
      -Convolution28_Output_0
      -Convolution110_Output_0
      -Times212_Output_0
      */

      /* This wont work as it is. For a given node, their inputs come from
      three different sources.
      1)Existing initializers in the model: model->graph->initializer[i]->name
      2)Provided input to the model: inputs[i]->name
      3)Previously calculated outputs of a different node.

      1) and 2) can be found with the existing code
      3) Is a bit more tricky and I think we would need a table that stores
      the pointer to that TensorProto
      */
      i->A = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      i->B = searchTensorProtoByName(model,
                                    inputs,
                                    nInputs,
                                    model->graph->node[nodeIdx]->input[1]);


      operator__onnx__add__context *c;
      c->in = i;
      c->out = o;
      c->attr = a;
      c.run = operator_add;

      all_op_context[nodeIdx] = c;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      /*...*/

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      /*...*/

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Maxpool")){
      /*...*/

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      /*...*/

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Matmul")){
      /*...*/
    }
  }

  return all_op_context;
}

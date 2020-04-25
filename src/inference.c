#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pb/onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "inference.h"

struct output_tensors* tensor_table;

void inference(struct operator__context** all_op_context, int n_nodes)
{
  TRACE_LEVEL0("Calling inference\n");

  // Iterate all nodes in the graph
  for (int nodeIdx = 0; nodeIdx < n_nodes; nodeIdx++)
  {
    printf("Calculating node = %d\n", nodeIdx);
    all_op_context[nodeIdx]->operator(all_op_context[nodeIdx]);
  }
}

/* Called once the model is loaded. Resolved each operator to a its function
and popualtes all structures with the inputs */
struct operator__context** resolve_check_get_input_and_attr(
                                     Onnx__ModelProto *model,
                                     Onnx__TensorProto **inputs,
                                     int nInputs)
{
  // Not sure about this malloc
  struct operator__context **all_op_context = malloc(sizeof(struct operator__context) * model->graph->n_node);

  tensor_table = malloc(sizeof(*tensor_table));
  tensor_table->n_tensors = 0;

  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    /* New idea prototyping, just a proof of concept */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      printf("Operator at %d is Add\n", nodeIdx);
      struct operator__onnx__add__input *i = malloc(sizeof(*i));
      struct operator__onnx__add__output *o = malloc(sizeof(*o));
      struct operator__onnx__add__attribute *a = malloc(sizeof(*a));

      i->A = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      i->B = searchTensorProtoByName(model,
                                    inputs,
                                    nInputs,
                                    model->graph->node[nodeIdx]->input[1]);

      //No attr


      struct operator__onnx__add__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->C = malloc(sizeof(*c->out->C));
      c->out->C->name = malloc(sizeof(char) * 40);
      strcpy(c->out->C->name, model->graph->node[nodeIdx]->output[0]);

      // Resolver operator
      c->operator = operator_add;

      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->C;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;

      all_op_context[nodeIdx] = (struct operator__context*)c;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      printf("Operator at %d is Conv\n", nodeIdx);
      struct operator__onnx__conv__input *i = malloc(sizeof(*i));
      struct operator__onnx__conv__output *o = malloc(sizeof(*o));
      struct operator__onnx__conv__attribute *a = malloc(sizeof(*a));
      i->X = NULL;
      i->W = NULL;
      i->B = NULL;
      /* Quick workaround to avoid accessing an element out of bound */
      if (model->graph->node[nodeIdx]->n_input >= 1){
        i->X = searchTensorProtoByName(model,
                                       inputs,
                                       nInputs,
                                       model->graph->node[nodeIdx]->input[0]);
      }
      if (model->graph->node[nodeIdx]->n_input >= 2){
        i->W = searchTensorProtoByName(model,
                                      inputs,
                                      nInputs,
                                      model->graph->node[nodeIdx]->input[1]);
      }
      if (model->graph->node[nodeIdx]->n_input >= 3){
        i->B = searchTensorProtoByName(model,
                                       inputs,
                                       nInputs,
                                       model->graph->node[nodeIdx]->input[2]);
      }

      // Attributes. NULL is returned if not found.
      a->auto_pad = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "auto_pad");
      a->dilations = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "dilations");
      a->group = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "group");
      a->kernel_shape = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "kernel_shape");
      a->pads = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "pads");
      a->strides = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "strides");

      struct operator__onnx__conv__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

      // Resolver operator
      c->operator = operator_conv;

      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->Y;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;

      all_op_context[nodeIdx] = (struct operator__context*)c;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      printf("Operator at %d is Relu\n", nodeIdx);
      struct operator__onnx__relu__input *i = malloc(sizeof(*i));
      struct operator__onnx__relu__output *o = malloc(sizeof(*o));
      struct operator__onnx__relu__attribute *a = malloc(sizeof(*a));

      // fixed n of i/o
      i->X = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      // not attr

      struct operator__onnx__relu__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

      // Resolver operator
      c->operator = operator_relu;


      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->Y;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;

      all_op_context[nodeIdx] = (struct operator__context*)c;


    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MaxPool")){
      printf("Operator at %d is Maxpool\n", nodeIdx);
      struct operator__onnx__maxpool__input *i = malloc(sizeof(*i));
      struct operator__onnx__maxpool__output *o = malloc(sizeof(*o));
      struct operator__onnx__maxpool__attribute *a = malloc(sizeof(*a));

      i->X = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      // Attributes. NULL is returned if not found.
      a->auto_pad = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "auto_pad");
      a->dilations = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "dilations");
      a->kernel_shape = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "kernel_shape");
      a->pads = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "pads");
      a->strides = searchAttributeNyName(model->graph->node[nodeIdx]->n_attribute, model->graph->node[nodeIdx]->attribute, "strides");


      struct operator__onnx__maxpool__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

      // Resolver operator
      c->operator = operator_maxpool;

      // TODO! For simplification only 1 output is stored. Maxpool has a second optional one. Not used in mnist.

      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->Y;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;

      all_op_context[nodeIdx] = (struct operator__context*)c;


    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      printf("Operator at %d is Reshape\n", nodeIdx);
      struct operator__onnx__reshape__input *i = malloc(sizeof(*i));
      struct operator__onnx__reshape__output *o = malloc(sizeof(*o));
      struct operator__onnx__reshape__attribute *a = malloc(sizeof(*a));

      // fixed n of inputs
      i->data = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      i->shape = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[1]);

      // no attr

      struct operator__onnx__reshape__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->reshaped = malloc(sizeof(*c->out->reshaped));
      c->out->reshaped->name = malloc(sizeof(char) * 40);
      strcpy(c->out->reshaped->name, model->graph->node[nodeIdx]->output[0]);

      // Resolver operator
      c->operator = operator_reshape;


      int tensor_idx = tensor_table->n_tensors;
      tensor_table->named_tensors[tensor_idx].tensor = &c->out->reshaped;
      strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
      tensor_table->n_tensors++;

      all_op_context[nodeIdx] = (struct operator__context*)c;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MatMul")){
      printf("Operator at %d is Matmul\n", nodeIdx);
      struct operator__onnx__matmul__input *i = malloc(sizeof(*i));
      struct operator__onnx__matmul__output *o = malloc(sizeof(*o));
      struct operator__onnx__matmul__attribute *a = malloc(sizeof(*a));

      // fixed n of inputs
      i->A = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);
     //
      i->B = searchTensorProtoByName(model,
                                    inputs,
                                    nInputs,
                                    model->graph->node[nodeIdx]->input[1]);

     // no attr

     struct operator__onnx__matmul__context *c = malloc(sizeof(*c));
     c->in = i;
     c->out = o;
     c->attr = a;
     c->out->Y = malloc(sizeof(*c->out->Y));
     c->out->Y->name = malloc(sizeof(char) * 40);
     strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

     // Resolver operator
     c->operator = operator_matmul;


     int tensor_idx = tensor_table->n_tensors;
     tensor_table->named_tensors[tensor_idx].tensor = &c->out->Y;
     strcpy(tensor_table->named_tensors[tensor_idx].name, model->graph->node[nodeIdx]->output[0]);
     tensor_table->n_tensors++;

     all_op_context[nodeIdx] = (struct operator__context*)c;

    }
  }

  return all_op_context;
}

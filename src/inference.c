#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pb/onnx.pb-c.h"
#include "utils.h"
#include "trace.h"
#include "inference.h"

int _outputIdx = 0;
Onnx__TensorProto *_outputs[MAX_NUM_OF_OUTPUTS] = {};

Onnx__TensorProto** lazy_outputs_mapping_tensors[MY_TABLE_SIZE] = {};

char lazy_output_mapping_names[MY_TABLE_SIZE][MAX_STRING_SIZE];

operator__context** all_op_context;


Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  //int error = 0;

  /* Dirty trick to allow multiple runs. There is a memory leak for sure */
  _outputIdx = 0;
  TRACE_LEVEL0("Calling inference\n");
  TRACE_LEVEL0("The graph has nodes=%zu\n", model->graph->n_node);

  printf("Resolving and getting inputs\n");
  all_op_context = resolve_check_get_input_and_attr(model,
                                                    inputs,
                                                    nInputs);
  printf("Done resolving\n");

  printf("\n\n\nRunning inference\n");
  // Iterate all nodes in the graph
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    printf("Calculating node = %d\n", nodeIdx);
    /* All this if will be replaced. This job will be done by the resoler */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      operator_add(all_op_context[nodeIdx]);
    }if (!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      operator_conv(all_op_context[nodeIdx]);
    }
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      operator_relu(all_op_context[nodeIdx]);
    }
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "MaxPool")){
      operator_maxpool(all_op_context[nodeIdx]);
    }
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      operator_reshape(all_op_context[nodeIdx]);
    }
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "MatMul")){
      operator_matmul(all_op_context[nodeIdx]);
    }

    Onnx__TensorProto *output = *lazy_outputs_mapping_tensors[nodeIdx];
    printf("Debug output %d name %s\n", nodeIdx, output->name);
    for (int i = 0; i < 5; i++){
      //printf("Printing some values [%d] = %f\n", i, output->float_data[i]);
    }
  }



  // TODO. Kept from legacy
  return _outputs;
}

/* Called once the model is loaded. Resolved each operator to a its function
and popualtes all structures with the inputs */
operator__context** resolve_check_get_input_and_attr(
                                     Onnx__ModelProto *model,
                                     Onnx__TensorProto **inputs,
                                     int nInputs)
{
  // Not sure about this malloc
  operator__context **all_op_context = malloc(sizeof(operator__context) * model->graph->n_node);
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    /* New idea prototyping, just a proof of concept */
    if (!strcmp(model->graph->node[nodeIdx]->op_type, "Add")){
      printf("Operator at %d is Add\n", nodeIdx);
      operator__onnx__add__input *i = malloc(sizeof(*i));
      operator__onnx__add__output *o = malloc(sizeof(*o));
      operator__onnx__add__attribute *a = malloc(sizeof(*a));

      i->A = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      i->B = searchTensorProtoByName(model,
                                    inputs,
                                    nInputs,
                                    model->graph->node[nodeIdx]->input[1]);

      //No attr


      operator__onnx__add__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->C = malloc(sizeof(*c->out->C));
      c->out->C->name = malloc(sizeof(char) * 40);
      strcpy(c->out->C->name, model->graph->node[nodeIdx]->output[0]);


      lazy_outputs_mapping_tensors[_outputIdx] = &c->out->C;
      strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
      _outputIdx++;

      all_op_context[nodeIdx] = (operator__context*)c;
    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Conv")){
      printf("Operator at %d is Conv\n", nodeIdx);
      operator__onnx__conv__input *i = malloc(sizeof(*i));
      operator__onnx__conv__output *o = malloc(sizeof(*o));
      operator__onnx__conv__attribute *a = malloc(sizeof(*a));
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

      operator__onnx__conv__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

      lazy_outputs_mapping_tensors[_outputIdx] = &c->out->Y;
      strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
      _outputIdx++;

      all_op_context[nodeIdx] = (operator__context*)c;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Relu")){
      printf("Operator at %d is Relu\n", nodeIdx);
      operator__onnx__relu__input *i = malloc(sizeof(*i));
      operator__onnx__relu__output *o = malloc(sizeof(*o));
      operator__onnx__relu__attribute *a = malloc(sizeof(*a));

      // fixed n of i/o
      i->X = searchTensorProtoByName(model,
                                     inputs,
                                     nInputs,
                                     model->graph->node[nodeIdx]->input[0]);

      // not attr

      operator__onnx__relu__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);


      lazy_outputs_mapping_tensors[_outputIdx] = &c->out->Y;
      strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
      _outputIdx++;

      all_op_context[nodeIdx] = (operator__context*)c;


    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MaxPool")){
      printf("Operator at %d is Maxpool\n", nodeIdx);
      operator__onnx__maxpool__input *i = malloc(sizeof(*i));
      operator__onnx__maxpool__output *o = malloc(sizeof(*o));
      operator__onnx__maxpool__attribute *a = malloc(sizeof(*a));

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


      operator__onnx__maxpool__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->Y = malloc(sizeof(*c->out->Y));
      c->out->Y->name = malloc(sizeof(char) * 40);
      strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);

      // TODO! For simplification only 1 output is stored. Maxpool has a second optional one. Not used in mnist.

      lazy_outputs_mapping_tensors[_outputIdx] = &c->out->Y;
      strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
      _outputIdx++;

      all_op_context[nodeIdx] = (operator__context*)c;


    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "Reshape")){
      printf("Operator at %d is Reshape\n", nodeIdx);
      operator__onnx__reshape__input *i = malloc(sizeof(*i));
      operator__onnx__reshape__output *o = malloc(sizeof(*o));
      operator__onnx__reshape__attribute *a = malloc(sizeof(*a));

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

      operator__onnx__reshape__context *c = malloc(sizeof(*c));
      c->in = i;
      c->out = o;
      c->attr = a;
      c->out->reshaped = malloc(sizeof(*c->out->reshaped));
      c->out->reshaped->name = malloc(sizeof(char) * 40);
      strcpy(c->out->reshaped->name, model->graph->node[nodeIdx]->output[0]);


      lazy_outputs_mapping_tensors[_outputIdx] = &c->out->reshaped;
      strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
      _outputIdx++;

      all_op_context[nodeIdx] = (operator__context*)c;

    }else if(!strcmp(model->graph->node[nodeIdx]->op_type, "MatMul")){
      printf("Operator at %d is Matmul\n", nodeIdx);
      operator__onnx__matmul__input *i = malloc(sizeof(*i));
      operator__onnx__matmul__output *o = malloc(sizeof(*o));
      operator__onnx__matmul__attribute *a = malloc(sizeof(*a));

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

     operator__onnx__matmul__context *c = malloc(sizeof(*c));
     c->in = i;
     c->out = o;
     c->attr = a;
     c->out->Y = malloc(sizeof(*c->out->Y));
     c->out->Y->name = malloc(sizeof(char) * 40);
     strcpy(c->out->Y->name, model->graph->node[nodeIdx]->output[0]);


     lazy_outputs_mapping_tensors[_outputIdx] = &c->out->Y;
     strcpy(lazy_output_mapping_names[_outputIdx], model->graph->node[nodeIdx]->output[0]);
     _outputIdx++;

     all_op_context[nodeIdx] = (operator__context*)c;

    }
  }

  return all_op_context;
}

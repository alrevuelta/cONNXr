#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_debug.h"

void Debug_PrintArray(float *array, int m, int n)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("array[%d][%d]=%f\n", i, j, array[i*m+j]);
    }
  }
}

void Debug_PrintModelInformation(Onnx__ModelProto *model)
{
  // Iterate the whole model and print it. TODO
  printf("model->producer_name %s\n", model->producer_name);
  printf("model->graph->n_node %zu\n", model->graph->n_node);
  printf("model->graph->name %s\n", model->graph->name);

  printf("model->graph->n_initializer %zu\n", model->graph->n_initializer);
  for (int n_init = 0; n_init < model->graph->n_initializer; n_init++)
  {
    printf("model->graph->initializer[%d] %s\n", n_init, model->graph->initializer[n_init]->name);
  }

  // input/output data
  printf("model->graph->n_input %zu\n", model->graph->n_input);
  printf("model->graph->n_output %zu\n", model->graph->n_output);
  printf("model->graph->input[0]->name %s\n", model->graph->input[0]->name);
  printf("model->graph->output[0]->name %s\n", model->graph->output[0]->name);

  printf("model->graph->input[0]->type->tensor_type->shape->n_dim %zu\n", model->graph->input[0]->type->tensor_type->shape->n_dim);
  // Index 0 has a random value and 1 is fine?
  printf("model->graph->input[0]->type->tensor_type->shape->dim[0]->dim_value %lld\n", model->graph->input[0]->type->tensor_type->shape->dim[0]->dim_value);
  printf("model->graph->input[0]->type->tensor_type->shape->dim[1]->dim_value %lld\n", model->graph->input[0]->type->tensor_type->shape->dim[1]->dim_value);

  printf("model->graph->output[0]->type->tensor_type->shape->n_dim %zu\n", model->graph->output[0]->type->tensor_type->shape->n_dim);
  printf("model->graph->output[0]->type->tensor_type->shape->dim[0]->dim_value %lld\n", model->graph->output[0]->type->tensor_type->shape->dim[0]->dim_value);
  printf("model->graph->output[0]->type->tensor_type->shape->dim[1]->dim_value %lld\n", model->graph->output[0]->type->tensor_type->shape->dim[1]->dim_value);
  /*
  for (int i = 0; i < model->graph->output[0]->type->tensor_type->shape->n_dim; i++)
  {
  printf("model->graph->output[0]->type->tensor_type->shape->n_dim %zu\n", model->graph->output[0]->type->tensor_type->shape->n_dim);
  printf("model->graph->output[0]->type->tensor_type->shape->dim[%d]->dim_value %lld\n", i, model->graph->output[0]->type->tensor_type->shape->dim[i]->dim_value);
  printf("model->graph->output[0]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, model->graph->output[0]->type->tensor_type->shape->dim[i]->dim_param);
}*/

for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
{
  printf("model->graph->node[%d]->n_input %zu\n", nodeIdx, model->graph->node[nodeIdx]->n_input);
  printf("model->graph->node[%d]->n_output %zu\n", nodeIdx, model->graph->node[nodeIdx]->n_output);
  printf("model->graph->node[%d]->name %s\n", nodeIdx, model->graph->node[nodeIdx]->name);
  printf("model->graph->node[%d]->op_type %s\n", nodeIdx, model->graph->node[nodeIdx]->op_type);

  char *operation = model->graph->node[nodeIdx]->op_type;
  printf("model->graph->node[nodeIdx]->n_input %zu\n", model->graph->node[nodeIdx]->n_input);
  int numInputs = model->graph->node[nodeIdx]->n_input;

}

// initizalizer
/*
printf("n_dims %zu\n", tensor->n_dims);
printf("n_float_data %zu\n", tensor->n_float_data);
printf("dims %lld\n", tensor->dims[dim]);
*/
}

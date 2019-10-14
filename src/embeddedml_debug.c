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
  //--------------------------------------------------------------------------//
  // MODEL
  //--------------------------------------------------------------------------//
  printf("model->producer_name %s\n", model->producer_name);
  printf("model->producer_version %s\n", model->producer_version);
  printf("model->n_opset_import %zu\n", model->n_opset_import);
  for (int i = 0; i < model->n_opset_import; i++) {
    printf("model->opset_import[%d]->domain %s\n", i, model->opset_import[i]->domain);
  }

  //--------------------------------------------------------------------------//
  // GRAPH
  //--------------------------------------------------------------------------//
  printf("model->graph->name %s\n", model->graph->name);
  printf("model->graph->n_node %zu\n", model->graph->n_node);
  printf("model->graph->n_initializer %zu\n", model->graph->n_initializer);
  for (int n_init = 0; n_init < model->graph->n_initializer; n_init++)
  {
    printf("model->graph->initializer[%d] %s\n", n_init, model->graph->initializer[n_init]->name);
  }

  // input/output data
  printf("model->graph->n_input %zu\n", model->graph->n_input);
  printf("model->graph->n_output %zu\n", model->graph->n_output);

  for (int i = 0; i < model->graph->n_input; i++) {
    printf("model->graph->input[%d]->name %s\n", i, model->graph->input[i]->name);
    printf("model->graph->input[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->input[i]->type->tensor_type->has_elem_type);
    printf("model->graph->input[%d]->type->tensor_type->elem_type %d\n", i, model->graph->input[i]->type->tensor_type->elem_type);
    printf("model->graph->input[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->input[i]->type->tensor_type->shape->n_dim);

    // TODO With some models this crashes
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      printf("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          printf("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          printf("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->dim_value %lld\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          printf("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;
      }
      //printf("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->denotation);
    }

  }
  /*
  for (int i = 0; i < model->graph->n_output; i++) {
    printf("model->graph->output[%d]->name %s\n", i, model->graph->output[i]->name);
    printf("model->graph->output[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->output[i]->type->tensor_type->has_elem_type);
    printf("model->graph->output[%d]->type->tensor_type->elem_type %d\n", i, model->graph->output[i]->type->tensor_type->elem_type);
    printf("model->graph->output[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->output[i]->type->tensor_type->shape->n_dim);
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      printf("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          printf("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          printf("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_value %lld\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          printf("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;
      }
      //printf("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->denotation);
    }
  }*/

  //--------------------------------------------------------------------------//
  // NODES
  //--------------------------------------------------------------------------//
  for (int i = 0; i < model->graph->n_node; i++)
  {
    printf("model->graph->node[%d]->n_input %zu\n", i, model->graph->node[i]->n_input);
    for (int j = 0; j < model->graph->node[i]->n_input; j++) {
      printf("model->graph->node[%d]->input[%d] %s\n", i, j, model->graph->node[i]->input[j]);
    }
    printf("model->graph->node[%d]->n_output %zu\n", i, model->graph->node[i]->n_output);
    for (int j = 0; j < model->graph->node[i]->n_output; j++) {
      printf("model->graph->node[%d]->output[%d] %s\n", i, j, model->graph->node[i]->output[j]);
    }
    printf("model->graph->node[%d]->name %s\n", i, model->graph->node[i]->name);
    printf("model->graph->node[%d]->op_type %s\n", i, model->graph->node[i]->op_type);
  }
}

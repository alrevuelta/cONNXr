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
        case _ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_IS_INT_SIZE:
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

    printf("model->graph->node[%d]->n_attribute %zu\n", i, model->graph->node[i]->n_attribute);
    for (int j = 0; j < model->graph->node[i]->n_attribute; j++)
    {
      // Check AttributeProto structure for more parameters
      printf("model->graph->node[%d]->attribute[%d]->name %s\n", i, j, model->graph->node[i]->attribute[j]->name);
    }
  }
}

void Debug_PrintTensorProto(Onnx__TensorProto *tp)
{
  printf("ndims = %zu\n", tp->n_dims);
  for (int i = 0; i < tp->n_dims; i++)
  {
    printf("dims[%d]=%lld\n", i, tp->dims[i]);
  }
  printf("has_data_type = %d\n", tp->has_data_type);
  printf("data_type = %d\n", tp->data_type);

  // TODO segment

  printf("n_float_data = %zu\n", tp->n_float_data);

  // Print float_data if needed

  printf("n_int32_data = %zu\n", tp->n_int32_data);

  // Print int32_data if needed

  printf("n_string_data = %zu\n", tp->n_string_data);

  // Print string_data if needed

  printf("n_int64_data = %zu\n", tp->n_int64_data);

  // Print int64_data if needed

  printf("name = %s\n", tp->name);
  printf("docstring = %s\n", tp->doc_string);

  printf("has_raw_data = %d\n", tp->has_raw_data);
  if (tp->has_raw_data)
  {
    printf("raw_data->len = %zu\n", tp->raw_data.len);

    // TODO 4 is hardcoded for a FLOAT case
    // According to doc this is little endian
    // Number are stored according to IEEE 754
    for (int i = 0; i < tp->raw_data.len; i+=4)
    {
      printf("merge the following i=%d: %d, %d, %d, %d\n",
                        i,
                        tp->raw_data.data[i],
                        tp->raw_data.data[i+1],
                        tp->raw_data.data[i+2],
                        tp->raw_data.data[i+3]);
      // Once float is 4 bytes.
      float unserNum = *(float *)&tp->raw_data.data[i];
      printf("unserialized number = %f\n", unserNum);
    }
  }

  // Print has_data_location if needed

  // Print data_location if needed

  printf("n_double_data = %zu\n", tp->n_double_data);

  // Print double_data if needed

  printf("n_uint64_data = %zu\n", tp->n_uint64_data);

  // Print uint64_data if needed

}

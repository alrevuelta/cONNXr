#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pb/onnx.pb-c.h"
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

void debug_print_attributes(size_t n_attribute, Onnx__AttributeProto **attribute)
{
  printf("n_attribute %zu\n", n_attribute);
  for (int j = 0; j < n_attribute; j++)
  {
    // Check AttributeProto structure for more parameters
    printf("attribute[%d]->name %s\n", j, attribute[j]->name);

    printf("attribute[%d]->has_type %d\n", j, attribute[j]->has_type);
    printf("attribute[%d]->type %d\n", j, attribute[j]->type);

    printf("attribute[%d]->has_f %d\n", j, attribute[j]->has_f);
    printf("attribute[%d]->has_i %d\n", j, attribute[j]->has_i);
    printf("attribute[%d]->has_s %d\n", j, attribute[j]->has_s);

    printf("attribute[%d]->n_floats %zu\n", j, attribute[j]->n_floats);
    for (int k = 0; k < attribute[j]->n_floats; k++)
    {
      printf("attribute[%d]->floats[%d] %f\n", j, k, attribute[j]->floats[k]);
    }

    printf("attribute[%d]->n_ints %zu\n", j, attribute[j]->n_ints);
    for (int k = 0; k < attribute[j]->n_ints; k++)
    {
      printf("attribute[%d]->ints[%d] %lld\n", j, k, attribute[j]->ints[k]);
    }

    printf("attribute[%d]->n_strings %zu\n", j, attribute[j]->n_strings);
    for (int k = 0; k < attribute[j]->n_strings; k++)
    {
      // Type is ProtobufCBinaryData
      //printf("attribute[%d]->strings[%d] %f\n", j, k, attribute[j]->string[k]);
    }

    printf("attribute[%d]->n_tensors %zu\n", j, attribute[j]->n_tensors);


    printf("attribute[%d]->n_graphs %zu\n", j, attribute[j]->n_graphs);
    printf("attribute[%d]->n_sparse_tensors %zu\n", j, attribute[j]->n_sparse_tensors);

    /*
    ProtobufCMessage base;
    char *name;
    char *ref_attr_name;
    char *doc_string;
    protobuf_c_boolean has_type;
    Onnx__AttributeProto__AttributeType type;
    //Exactly ONE of the following fields must be present for this version of the IR
    protobuf_c_boolean has_f;
    float f;
    protobuf_c_boolean has_i;
    int64_t i;
    protobuf_c_boolean has_s;
    ProtobufCBinaryData s;
    Onnx__TensorProto *t;
    Onnx__GraphProto *g;
    Onnx__SparseTensorProto *sparse_tensor;
    size_t n_floats;
    float *floats;
    size_t n_ints;
    int64_t *ints;
    size_t n_strings;
    ProtobufCBinaryData *strings;
    size_t n_tensors;
    Onnx__TensorProto **tensors;
    size_t n_graphs;
    Onnx__GraphProto **graphs;
    size_t n_sparse_tensors;
    Onnx__SparseTensorProto **sparse_tensors;*/
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

    debug_print_attributes(model->graph->node[i]->n_attribute, model->graph->node[i]->attribute);
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

    // print raw data. convert to data_type if needed like it is done in
    // utils.c
    for (int i = 0; i < tp->raw_data.len; i++)
    {
      printf("tp->raw_data.data[%d] = %d\n", i, tp->raw_data.data[i]);
    }
  }

  // Print has_data_location if needed

  // Print data_location if needed

  printf("n_double_data = %zu\n", tp->n_double_data);

  // Print double_data if needed

  printf("n_uint64_data = %zu\n", tp->n_uint64_data);

  // Print uint64_data if needed

}

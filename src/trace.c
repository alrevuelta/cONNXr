#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "onnx.pb-c.h"
#include "trace.h"

char data_types_string[][20] = {
                         "UNDEFINED",  /* ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED = 0 */
                         "FLOAT",      /* ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT = 1     */
                         "UINT8",      /* ONNX__TENSOR_PROTO__DATA_TYPE__UINT8 = 2     */
                         "INT8",       /* ONNX__TENSOR_PROTO__DATA_TYPE__INT8 = 3      */
                         "UINT16",     /* ONNX__TENSOR_PROTO__DATA_TYPE__UINT16 = 4    */
                         "INT16",      /* ONNX__TENSOR_PROTO__DATA_TYPE__INT16 = 5     */
                         "INT32",      /* ONNX__TENSOR_PROTO__DATA_TYPE__INT32 = 6     */
                         "INT64",      /* ONNX__TENSOR_PROTO__DATA_TYPE__INT64 = 7     */
                         "STRING",     /* ONNX__TENSOR_PROTO__DATA_TYPE__STRING = 8    */
                         "BOOL",       /* ONNX__TENSOR_PROTO__DATA_TYPE__BOOL = 9      */
                         "FLOT16",     /* ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 = 10  */
                         "DOUBLE",     /* ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE = 11   */
                         "UINT32",     /* ONNX__TENSOR_PROTO__DATA_TYPE__UINT32 = 12   */
                         "UINT64",     /* ONNX__TENSOR_PROTO__DATA_TYPE__UINT64 = 13   */
                         "COMPLEX64",  /* ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64 = 14  */
                         "COMPLEX128", /* ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128 = 15 */
                         "BFLOAT16"    /* ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16 = 16   */
                     };

void Debug_PrintArray(float *array, int m, int n)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      TRACE_LEVEL0("array[%d][%d]=%f\n", i, j, array[i*m+j]);
    }
  }
}

void debug_print_attributes( size_t n_attribute,  Onnx__AttributeProto **attribute)
{
  TRACE_LEVEL0("n_attribute %zu\n", n_attribute);
  for (int j = 0; j < n_attribute; j++)
  {
    // Check AttributeProto structure for more parameters
    TRACE_LEVEL0("attribute[%d]->name %s\n", j, attribute[j]->name);

    TRACE_LEVEL0("attribute[%d]->has_type %d\n", j, attribute[j]->has_type);
    TRACE_LEVEL0("attribute[%d]->type %d\n", j, attribute[j]->type);

    TRACE_LEVEL0("attribute[%d]->has_f %d\n", j, attribute[j]->has_f);
    //print f

    TRACE_LEVEL0("attribute[%d]->has_i %d\n", j, attribute[j]->has_i);
    // print i

    TRACE_LEVEL0("attribute[%d]->has_s %d\n", j, attribute[j]->has_s);
    if (attribute[j]->has_s) {
      // This has s.data and s.len
      TRACE_LEVEL0("attribute[%d]->s %s\n", j, attribute[j]->s.data);
    }


    TRACE_LEVEL0("attribute[%d]->n_floats %zu\n", j, attribute[j]->n_floats);
    for (int k = 0; k < attribute[j]->n_floats; k++)
    {
      TRACE_LEVEL0("attribute[%d]->floats[%d] %f\n", j, k, attribute[j]->floats[k]);
    }

    TRACE_LEVEL0("attribute[%d]->n_ints %zu\n", j, attribute[j]->n_ints);
    for (int k = 0; k < attribute[j]->n_ints; k++)
    {
      TRACE_LEVEL0("attribute[%d]->ints[%d] %" PRId64 "\n", j, k, attribute[j]->ints[k]);
    }

    TRACE_LEVEL0("attribute[%d]->n_strings %zu\n", j, attribute[j]->n_strings);
    for (int k = 0; k < attribute[j]->n_strings; k++)
    {
      // Type is ProtobufCBinaryData
      //TRACE_LEVEL0("attribute[%d]->strings[%d] %f\n", j, k, attribute[j]->string[k]);
    }

    TRACE_LEVEL0("attribute[%d]->n_tensors %zu\n", j, attribute[j]->n_tensors);


    TRACE_LEVEL0("attribute[%d]->n_graphs %zu\n", j, attribute[j]->n_graphs);
    TRACE_LEVEL0("attribute[%d]->n_sparse_tensors %zu\n", j, attribute[j]->n_sparse_tensors);

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

void debug_print_dims(size_t n_dims, int64_t *dims)
{
  TRACE_LEVEL0("n_dims=%zu\n", n_dims);
  for (int i = 0; i < n_dims; i++){
    TRACE_LEVEL0("dims[%d]=%" PRId64 "\n", i, dims[i]);
  }
}

void debug_prettyprint_tensorproto(Onnx__TensorProto *tp)
{
}

void Debug_PrintModelInformation( Onnx__ModelProto *model)
{
  //--------------------------------------------------------------------------//
  // MODEL
  //--------------------------------------------------------------------------//
  TRACE_LEVEL0("model->producer_name %s\n", model->producer_name);
  TRACE_LEVEL0("model->producer_version %s\n", model->producer_version);
  TRACE_LEVEL0("model->n_opset_import %zu\n", model->n_opset_import);
  for (int i = 0; i < model->n_opset_import; i++) {
    TRACE_LEVEL0("model->opset_import[%d]->domain %s\n", i, model->opset_import[i]->domain);
  }

  //--------------------------------------------------------------------------//
  // GRAPH
  //--------------------------------------------------------------------------//
  TRACE_LEVEL0("model->graph->name %s\n", model->graph->name);
  TRACE_LEVEL0("model->graph->n_node %zu\n", model->graph->n_node);
  TRACE_LEVEL0("model->graph->n_initializer %zu\n", model->graph->n_initializer);
  for (int n_init = 0; n_init < model->graph->n_initializer; n_init++)
  {
    TRACE_LEVEL0("model->graph->initializer[%d] %s\n", n_init, model->graph->initializer[n_init]->name);
    Debug_PrintTensorProto(model->graph->initializer[n_init]);
  }

  // input/output data
  TRACE_LEVEL0("model->graph->n_input %zu\n", model->graph->n_input);
  TRACE_LEVEL0("model->graph->n_output %zu\n", model->graph->n_output);

  for (int i = 0; i < model->graph->n_input; i++) {
    TRACE_LEVEL0("model->graph->input[%d]->name %s\n", i, model->graph->input[i]->name);
    TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->input[i]->type->tensor_type->has_elem_type);
    TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->elem_type %" PRId32 "\n", i, model->graph->input[i]->type->tensor_type->elem_type);
    TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->input[i]->type->tensor_type->shape->n_dim);

    // TODO With some models this crashes
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          TRACE_LEVEL0("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->dim_value %" PRId64 "\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;
        case _ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_IS_INT_SIZE:
          break;
      }
      //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->denotation);
    }

  }
  /*
  for (int i = 0; i < model->graph->n_output; i++) {
    TRACE_LEVEL0("model->graph->output[%d]->name %s\n", i, model->graph->output[i]->name);
    TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->output[i]->type->tensor_type->has_elem_type);
    TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->elem_type %d\n", i, model->graph->output[i]->type->tensor_type->elem_type);
    TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->output[i]->type->tensor_type->shape->n_dim);
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          TRACE_LEVEL0("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_value %lld\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;
      }
      //TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->denotation);
    }
  }*/

  //--------------------------------------------------------------------------//
  // NODES
  //--------------------------------------------------------------------------//
  for (int i = 0; i < model->graph->n_node; i++)
  {
    TRACE_LEVEL0("model->graph->node[%d]->n_input %zu\n", i, model->graph->node[i]->n_input);
    for (int j = 0; j < model->graph->node[i]->n_input; j++) {
      TRACE_LEVEL0("model->graph->node[%d]->input[%d] %s\n", i, j, model->graph->node[i]->input[j]);
    }
    TRACE_LEVEL0("model->graph->node[%d]->n_output %zu\n", i, model->graph->node[i]->n_output);
    for (int j = 0; j < model->graph->node[i]->n_output; j++) {
      TRACE_LEVEL0("model->graph->node[%d]->output[%d] %s\n", i, j, model->graph->node[i]->output[j]);
    }
    TRACE_LEVEL0("model->graph->node[%d]->name %s\n", i, model->graph->node[i]->name);
    TRACE_LEVEL0("model->graph->node[%d]->op_type %s\n", i, model->graph->node[i]->op_type);

    debug_print_attributes(model->graph->node[i]->n_attribute, model->graph->node[i]->attribute);
  }
}

void Debug_PrintTensorProto(Onnx__TensorProto *tp)
{
  TRACE_LEVEL0("Printing tensorProto with name %s\n", tp->name);
  TRACE_LEVEL0("ndims = %zu\n", tp->n_dims);
  for (int i = 0; i < tp->n_dims; i++)
  {
    TRACE_LEVEL0("dims[%d]=%" PRId64 "\n", i, tp->dims[i]);
  }
  TRACE_LEVEL0("has_data_type = %d\n", tp->has_data_type);
  TRACE_LEVEL0("data_type = %" PRId32 "\n", tp->data_type);

  // TODO segment

  TRACE_LEVEL0("n_float_data = %zu\n", tp->n_float_data);

  // Print float_data if needed
  // Plot just first 10 values
  int float_to_print = tp->n_float_data > 10 ? 10 : tp->n_float_data;
  for (int i = 0; i < float_to_print; i++) {
    TRACE_LEVEL0("float_data[%d] = %f\n", i, tp->float_data[i]);
  }

  TRACE_LEVEL0("n_int32_data = %zu\n", tp->n_int32_data);

  // Print int32_data if needed

  TRACE_LEVEL0("n_string_data = %zu\n", tp->n_string_data);

  // Print string_data if needed

  TRACE_LEVEL0("n_int64_data = %zu\n", tp->n_int64_data);

  // Print int64_data if needed

  TRACE_LEVEL0("name = %s\n", tp->name);
  TRACE_LEVEL0("docstring = %s\n", tp->doc_string);

  TRACE_LEVEL0("has_raw_data = %d\n", tp->has_raw_data);
  if (tp->has_raw_data)
  {
    TRACE_LEVEL0("raw_data->len = %zu\n", tp->raw_data.len);

    // print raw data. convert to data_type if needed like it is done in
    // utils.c
    for (int i = 0; i < tp->raw_data.len; i++)
    {
      TRACE_LEVEL0("tp->raw_data.data[%d] = %d\n", i, tp->raw_data.data[i]);
    }
  }

  // Print has_data_location if needed

  // Print data_location if needed

  TRACE_LEVEL0("n_double_data = %zu\n", tp->n_double_data);

  // Print double_data if needed

  TRACE_LEVEL0("n_uint64_data = %zu\n", tp->n_uint64_data);

  // Print uint64_data if needed

}



void debug_prettyprint_model(Onnx__ModelProto *model)
{
  TRACE_LEVEL0("");
  TRACE_LEVEL0("-----------------------------------------------\n");
  TRACE_LEVEL0("---------------Model information---------------\n");
  TRACE_LEVEL0("-----------------------------------------------\n");

  TRACE_LEVEL0("Model:\n");
  TRACE_LEVEL0("  model->producer_name %s\n", model->producer_name);
  TRACE_LEVEL0("  model->producer_version %s\n", model->producer_version);
  TRACE_LEVEL0("  model->n_opset_import %zu\n", model->n_opset_import);
  for (int i = 0; i < model->n_opset_import; i++) {
    TRACE_LEVEL0("  model->opset_import[%d]->domain %s\n", i, model->opset_import[i]->domain);
  }

  //--------------------------------------------------------------------------//
  // GRAPH
  //--------------------------------------------------------------------------//
  TRACE_LEVEL0("Graph:\n");
  TRACE_LEVEL0("  model->graph->name %s\n", model->graph->name);
  TRACE_LEVEL0("  model->graph->n_node %zu\n", model->graph->n_node);
  TRACE_LEVEL0("  model->graph->n_initializer %zu\n", model->graph->n_initializer);

  TRACE_LEVEL0("Initializers:\n");
  for (int n_init = 0; n_init < model->graph->n_initializer; n_init++)
  {
    TRACE_LEVEL0("  model->graph->initializer[%d] %s\t\t|", n_init,
                                                      model->graph->initializer[n_init]->name);

    for (int i = 0; i < model->graph->initializer[n_init]->n_dims; i++)
    {
      TRACE_LEVEL0("%" PRId64 " x ", model->graph->initializer[n_init]->dims[i]);
    }
    //print(" | has_data_type = %d\n", tp->has_data_type);
    TRACE_LEVEL0("\t| %s\n", data_types_string[model->graph->initializer[n_init]->data_type]);
  }

/*
  TRACE_LEVEL0("Nodes:\n");
  TRACE_LEVEL0("  model->graph->n_input %zu\n", model->graph->n_input);
  TRACE_LEVEL0("  model->graph->n_output %zu\n", model->graph->n_output);*/

/*
  for (int i = 0; i < model->graph->n_input; i++) {
    TRACE_LEVEL0("%zu %s | ", i, model->graph->input[i]->name);

    //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->input[i]->type->tensor_type->has_elem_type);
    //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->elem_type %d\n", i, model->graph->input[i]->type->tensor_type->elem_type);
    //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->input[i]->type->tensor_type->shape->n_dim);

    // TODO With some models this crashes
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->input[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          TRACE_LEVEL0("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          TRACE_LEVEL0("%lld x ", model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          TRACE_LEVEL0("TODO %s\n", model->graph->input[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;
        case _ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_IS_INT_SIZE:
          break;
      }
      //TRACE_LEVEL0("model->graph->input[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->input[i]->type->tensor_type->shape->dim[j]->denotation);
    }
    TRACE_LEVEL0("\n");

  }*/

  for (int i = 0; i < model->graph->n_output; i++) {
    //TRACE_LEVEL0("model->graph->output[%d]->name %s\n", i, model->graph->output[i]->name);
    //TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->has_elem_type %d\n", i, model->graph->output[i]->type->tensor_type->has_elem_type);
    //TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->elem_type %d\n", i, model->graph->output[i]->type->tensor_type->elem_type);
    //TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->n_dim %zu\n", i, model->graph->output[i]->type->tensor_type->shape->n_dim);
    /* Is this failing?
    for (int j = 0; j < model->graph->input[i]->type->tensor_type->shape->n_dim; j++) {

      TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->value_case %d\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case);
      switch(model->graph->output[i]->type->tensor_type->shape->dim[j]->value_case) {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
          TRACE_LEVEL0("Value not not set\n");
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
          TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_value %lld\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_value);
          break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
          TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->dim_param %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->dim_param);
          break;

      //TRACE_LEVEL0("model->graph->output[%d]->type->tensor_type->shape->dim[%d]->denotation %s\n", i, j, model->graph->output[i]->type->tensor_type->shape->dim[j]->denotation);
    }*/
  }


  /* TODO Experiment with the align
  https://stackoverflow.com/questions/35329208/aligning-columns-in-c-output
  */
  TRACE_LEVEL0("Nodes:\n");
  for (int i = 0; i < model->graph->n_node; i++)
  {
    TRACE_LEVEL0("  %s %20.20s n_input=%zu  n_output=%zu\n",
                                          model->graph->node[i]->op_type,
                                          model->graph->node[i]->name,
                                          model->graph->node[i]->n_input,
                                          model->graph->node[i]->n_output);
    //TRACE_LEVEL0("model->graph->node[%d]->op_type %s\n", i, model->graph->node[i]->op_type);
    //TRACE_LEVEL0("model->graph->node[%d]->n_input %zu\n", i, model->graph->node[i]->n_input);
    for (int j = 0; j < model->graph->node[i]->n_input; j++) {
      TRACE_LEVEL0("    input[%d] %s\n", j, model->graph->node[i]->input[j]);
    }
    for (int j = 0; j < model->graph->node[i]->n_output; j++) {
      TRACE_LEVEL0("    output[%d] %s\n", j, model->graph->node[i]->output[j]);
    }

    for (int j = 0; j < model->graph->node[i]->n_attribute; j++)
    {
      TRACE_LEVEL0("    attribute[%d]->name %s\n", j, model->graph->node[i]->attribute[j]->name);
      //TRACE_LEVEL0("    attribute[%d]->has_type %d\n", j, model->graph->node[i]->attribute[j]->has_type);
      //TRACE_LEVEL0("    attribute[%d]->type %d\n", j, model->graph->node[i]->attribute[j]->type);
    }
  }
}
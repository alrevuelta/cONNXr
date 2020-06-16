#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "trace.h"

// TODO Rethink this function?
Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model,
                                           Onnx__TensorProto **inputs,
                                           int nInputs,
                                           char *name)
{
  TRACE_LEVEL0("Searching for TensorProto with name=%s\n", name);

  // Search in initializers
  for (int initializer = 0; initializer < model->graph->n_initializer; initializer++)
  {
    if (!strcmp(model->graph->initializer[initializer]->name, name))
    {
      TRACE_LEVEL0("Found TensorProto in initializer list with name=%s\n", model->graph->initializer[initializer]->name);
      return model->graph->initializer[initializer];
    }
  }

  // Search in inputs to the model
  for (int inIdx = 0; inIdx < nInputs; inIdx++)
  {
    if (!strcmp(inputs[inIdx]->name, name))
    {
      TRACE_LEVEL0("Found TensorProto in inputs to de model with name=%s\n", inputs[inIdx]->name);
      return inputs[inIdx];
    }
  }

  // Search in new context. Only for outputs
  if (_populatedIdx != -1)
  {
    // Iterate all populated nodes
    for (int node_i = 0; node_i < _populatedIdx+1; node_i++)
    {
      for (int output_i = 0; output_i < all_context[node_i].onnx_node->n_output; output_i++){
        TRACE_LEVEL0("Searching %s, found %s\n", name, all_context[node_i].outputs[output_i]->name);
        if (!strcmp(all_context[node_i].outputs[output_i]->name, name))
        {
          TRACE_LEVEL0("Found TensorProto in outputs from new context name=%s\n", all_context[node_i].outputs[output_i]->name);
          return all_context[node_i].outputs[output_i];
        }
      }
    }
  }

  TRACE_LEVEL0("%s was not found anywhere, maybe you should worry\n", name);
  return NULL;
}

Onnx__TensorProto* searchInputByName(node_context *ctx,
                                     int index)
{
  // Just return null if we are accesing an optional parameters that is not present
  if (index > ctx->onnx_node->n_input-1)
  {
    return NULL;
  }

  for (int i = 0; i < ctx->onnx_node->n_input; i++)
  {
    printf("Searching inputs %s, %s\n", ctx->inputs[i]->name, ctx->onnx_node->input[index]);
    if (!strcmp(ctx->inputs[i]->name, ctx->onnx_node->input[index]))
    {
      return ctx->inputs[i];
    }
  }
  printf("%s not found\n", ctx->onnx_node->input[index]);
  return NULL;
}

Onnx__TensorProto* searchOutputByName(node_context *ctx,
                                      int index)
{
  // Just return null if we are accesing an optional parameters that is not present
  if (index > ctx->onnx_node->n_output-1)
  {
    return NULL;
  }

  for (int i = 0; i < ctx->onnx_node->n_output; i++)
  {
    if (!strcmp(ctx->outputs[i]->name, ctx->onnx_node->output[index]))
    {
      return ctx->outputs[i];
    }
  }
  printf("%s not found\n", ctx->onnx_node->output[index]);
  return NULL;
}

Onnx__AttributeProto* searchAttributeNyName( size_t n_attribute,
                                             Onnx__AttributeProto **attribute,
                                             char *name)
{
  for (int i = 0; i < n_attribute; i++){
    if (!strcmp(attribute[i]->name, name)){
      TRACE_LEVEL0("Attribute %s was found\n", name);
      return attribute[i];
    }
  }
  TRACE_LEVEL0("Attribute %s was NOT found\n", name);
  return NULL;
}

Onnx__ModelProto* openOnnxFile(char *fname){
  Onnx__ModelProto *model = NULL;

  FILE *fl = fopen(fname, "r");
  if (fl == NULL){
    TRACE_LEVEL0("File was not opened\n");
    return model;
  }

  fseek(fl, 0, SEEK_END);
  long len = ftell(fl);
  uint8_t *ret = malloc(len);
  fseek(fl, 0, SEEK_SET);
  fread(ret, 1, len, fl);
  fclose(fl);

  TRACE_LEVEL0("length of file is %ld\n", len);

  model = onnx__model_proto__unpack(NULL,len,ret);

  return model;
}

Onnx__TensorProto *openTensorProtoFile(char *fname){
  Onnx__TensorProto *model = NULL;

  TRACE_LEVEL0("Opening .pb file %s\n", fname);

  FILE *fl = fopen(fname, "r");
  if (fl == NULL){
    return model;
  }

  fseek(fl, 0, SEEK_END);
  long len = ftell(fl);
  uint8_t *ret = malloc(len);
  fseek(fl, 0, SEEK_SET);
  fread(ret, 1, len, fl);
  fclose(fl);

  TRACE_LEVEL0("length of file %ld\n", len);

  model = onnx__tensor_proto__unpack(NULL,len,ret);

  return model;
}

/*
Takes as an input a tensor with has_raw_data field TRUE, reads raw_data and
stores it into "formated" data in the corresponding field. Hardcoded for float
*/
int convertRawDataOfTensorProto(Onnx__TensorProto *tensor)
{
  if (tensor == NULL)
  {
    TRACE_LEVEL0("Tensor is null, break\n");
  }

  if (tensor->has_raw_data)
  {
    TRACE_LEVEL0("Tensor has raw data. Unserializing it\n");
    TRACE_LEVEL0("Tensor type = %d\n", tensor->data_type);

    switch(tensor->data_type)
    {
      case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      {
        tensor->n_float_data = tensor->raw_data.len/4;
        // todo use sizeof
        tensor->float_data = malloc(tensor->n_float_data * sizeof(float));
        for (int i = 0; i < tensor->raw_data.len; i+=4)
        {
          // Once float is 4 bytes.
          tensor->float_data[i/4] = *(float *)&tensor->raw_data.data[i];
          //TRACE_LEVEL0("tensor->float_data[%d] = %f", i/4, tensor->float_data[i/4]);
        }
      } break;
      /* TODO I think uint8 and 16/32 can be all merged since the type is the same*/
      case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
        tensor->n_int32_data = tensor->raw_data.len/sizeof(uint8_t);
        tensor->int32_data = malloc(tensor->n_int32_data * sizeof(int32_t));
        for (int i = 0; i < tensor->raw_data.len; i+=sizeof(uint8_t)){
          tensor->int32_data[i/sizeof(uint8_t)] = *(uint8_t *)&tensor->raw_data.data[i];
        }
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      {
        tensor->n_int64_data = tensor->raw_data.len/sizeof(int64_t);
        tensor->int64_data = malloc(tensor->n_int64_data * sizeof(int64_t));
        for (int i = 0; i < tensor->raw_data.len; i+=sizeof(int64_t))
        {
          tensor->int64_data[i/sizeof(int64_t)] = *(int64_t *)&tensor->raw_data.data[i];
          //TRACE_LEVEL0("tensor->int64_data[%lu] = %lld", i/sizeof(int64_t), tensor->int64_data[i/sizeof(int64_t)]);
        }
      } break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
        break;
      default:
        break;
    }

    // TODO is this allowed?
    free(tensor->raw_data.data);
    tensor->has_raw_data = 0;
    tensor->raw_data.len = 0;
  }
  else
  {
    TRACE_LEVEL0("Input tensor doesnt have raw_data, doing nothing\n");
  }

  return 0;
}

/* tp has already memory allocated for the struct. Maybe its
memory should be allocated here instead. This is a bit experimental TODO:*/
/* this is crap, remove. not used*/
void mallocTensorProto(Onnx__TensorProto *tp,
                       Onnx__TensorProto__DataType data_type,
                       size_t n_dims,
                       size_t n_data)
{

  tp->has_raw_data = 0;
  tp->n_external_data = 0;
  tp->has_data_location = 0;
  tp->n_dims = n_dims;
  tp->dims = malloc(n_dims * sizeof(int64_t));
  tp->name = malloc(MAX_CHAR_SIZE * sizeof(char));

  if (tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT ||
      tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64){
    tp->n_float_data = n_data;
    tp->float_data = malloc(n_data * sizeof(float));

  }else if (tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT32  ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT16  ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT8   ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UINT16 ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UINT8  ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__BOOL   ||
            tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16){
    tp->n_int32_data = n_data;
    tp->int32_data = malloc(n_data * sizeof(int32_t));

  }else if (tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT64){
    tp->n_int64_data = n_data;
    tp->int64_data = malloc(n_data * sizeof(int64_t));

  }else if(tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__STRING){
    /* TODO
    size_t n_string_data;
    ProtobufCBinaryData *string_data; */

  }else if(tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE ||
          tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128){
    tp->n_double_data = n_data;
    tp->double_data = malloc(n_data * sizeof(double));

  }else if(tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UINT64 ||
          tp->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__UINT32){
    tp->n_uint64_data = n_data;
    tp->uint64_data = malloc(n_data * sizeof(uint64_t));
  }
}

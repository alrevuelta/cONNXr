#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_utils.h"
#include "embeddedml_debug.h"
#include "embeddedml_inference.h"

Onnx__TensorProto* searchTensorProtoByName(Onnx__ModelProto *model,
                                           Onnx__TensorProto **inputs,
                                           int nInputs,
                                           char *name)
{
  DEBUG_PRINT("Searching for TensorProto with name=%s", name);
  Onnx__TensorProto *tensor = NULL;

  // Search in initializers
  for (int initializer = 0; initializer < model->graph->n_initializer; initializer++)
  {
    if (!strcmp(model->graph->initializer[initializer]->name, name))
    {
      tensor = model->graph->initializer[initializer];
      DEBUG_PRINT("Found TensorProto in initializer list with name=%s", model->graph->initializer[initializer]->name);
      break;
      // Use return instead. Once its found, exit the function
    }
  }

  // Search in inputs to the model
  for (int inIdx = 0; inIdx < nInputs; inIdx++)
  {
    if (!strcmp(inputs[inIdx]->name, name))
    {
      tensor = inputs[inIdx];
      DEBUG_PRINT("Found TensorProto in inputs to de model with name=%s", inputs[inIdx]->name);
      break;
      // Use return instead. Once its found, exit the function
    }
  }

  // Search in calculated outputs list
  for (int outputsIdx = 0; outputsIdx < _outputIdx; outputsIdx++)
  {
    if (!strcmp(_outputs[outputsIdx]->name, name))
    {
      tensor = _outputs[outputsIdx];
      DEBUG_PRINT("Found TensorProto in outputs list with name=%s", inputs[outputsIdx]->name);
      break;
      // Use return instead. Once its found, exit the function
    }
  }

  return tensor;
}

Onnx__ModelProto* openOnnxFile(char *fname){
  Onnx__ModelProto *model = NULL;

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

  DEBUG_PRINT("length of file is %ld", len);

  model = onnx__model_proto__unpack(NULL,len,ret);

  return model;
}

Onnx__TensorProto *openTensorProtoFile(char *fname){
  Onnx__TensorProto *model = NULL;

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

  DEBUG_PRINT("length of file %ld", len);

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
    DEBUG_PRINT("Tensor is null, break");
  }

  if (tensor->has_raw_data)
  {
    DEBUG_PRINT("Tensor has raw data. Unserializing it");
    DEBUG_PRINT("Tensor type = %d", tensor->data_type);

    switch(tensor->data_type)
    {
      case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
        break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      {
        tensor->n_float_data = tensor->raw_data.len/4;
        tensor->float_data = malloc(tensor->n_float_data * sizeof(float));
        for (int i = 0; i < tensor->raw_data.len; i+=4)
        {
          // Once float is 4 bytes.
          tensor->float_data[i/4] = *(float *)&tensor->raw_data.data[i];
          //DEBUG_PRINT("Writing %f", tensor->float_data[i/4]);
        }
      } break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
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
          DEBUG_PRINT("Writing %lld", tensor->int64_data[i/sizeof(int64_t)]);
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
    DEBUG_PRINT("Input tensor doesnt have raw_data, doing nothing");
  }

  return 0;
}

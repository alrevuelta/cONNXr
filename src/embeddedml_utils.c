#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_utils.h"
#include "embeddedml_debug.h"


Onnx__TensorProto* searchTensorForNode(Onnx__ModelProto *model, int nodeIdx)
{
  Onnx__TensorProto *tensor = NULL;
  for (int input = 0; input < model->graph->node[nodeIdx]->n_input; input++)
  {
    for (int initializer = 0; initializer < model->graph->n_initializer; initializer++)
    {
      if (!strcmp(model->graph->initializer[initializer]->name, model->graph->node[nodeIdx]->input[input]))
      {
        tensor = model->graph->initializer[initializer];
        break;
      }
    }
  }
  return tensor;
}

int getDimensionsOfTensor(Onnx__TensorProto *tensor)
{
  int totalDim = 1;
  for (int dim = 0; dim < tensor->n_dims; dim++)
  {
    totalDim *= tensor->dims[dim];
  }
  return totalDim;
}

Onnx__ModelProto *openOnnxFile(char *fname){
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

  printf("length of file is %ld\n", len);

  model = onnx__model_proto__unpack(NULL,len,ret);

  return model;
}

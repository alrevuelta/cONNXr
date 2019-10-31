#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_operators.h"
#include "embeddedml_outputs.h"
#include "embeddedml_debug.h"

void outputs_allocAllTensors() {
  // Do this dynamically in the future
/*
  int numOfTensorsToStore = 30;
  calculaterTensors = malloc(numOfTensorsToStore * sizeof(Onnx__TensorProto*))
  for (int i = 0; i < numOfTensorsToStore; i++) {
    calculaterTensors[i] = malloc(sizeof(Onnx__TensorProto));
  }*/
}
void outputs_freeAllTensors() {
  // free all dynamically allocated mem.
}

void outputs_allocateOneTensor(Onnx__TensorProto *tpToAllocate, int32_t data_type, int *dimensions, int nDims) {
  //Allocate memory for one tensor, given a data type and some dimensions.


}

Onnx__TensorProto *outputs_searchByName(char *name) {
  Onnx__TensorProto *tensor = NULL;
  // Iterate only the indexes that have content
  for (int i = 0; i < tensorIdx; i++) {
    if (!strcmp(calculaterTensors[i].name, name))
    {
      tensor = &calculaterTensors[i];
      DEBUG_PRINT("Found calculated tensor with name=%s", calculaterTensors[i].name);
      break;
    }
  }
  return tensor;
}
int outputs_addNewOutput(Onnx__TensorProto *tpToAdd) {
  if (tensorIdx < (MAX_TENSORS_BUFFER_SIZE - 1)) {
    tensorIdx++;
    calculaterTensors[tensorIdx] = *tpToAdd;
    return 0;
  } else {
    return 1;
  }
}

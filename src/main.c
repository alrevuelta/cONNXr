#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "embeddedml_debug.h"
#include "embeddedml_operators.h"
#include "embeddedml_opwrapper.h"
#include "embeddedml_utils.h"
#include "embeddedml_inference.h"

int main()
{
  Onnx__ModelProto *model = openOnnxFile("../models/mnist/model.onnx");
  if (model == NULL)
  {
    perror("Error when opening the onnx file\n");
    exit(-1);
  }

  DEBUG_PRINT("Debugging is on");

  Debug_PrintModelInformation(model);


  int inputDim = model->graph->input[0]->type->tensor_type->shape->dim[1]->dim_value;
  float *input = malloc(inputDim * sizeof(float));

  for (int i = 0; i < inputDim; i++)
  {
    // Just a random input
    input[i] = i;
  }

  int error = inferenceFloat(input, inputDim, model);

  if (error)
  {
    perror("There was an error during the inference\n");
    exit(-1);
  }

  // TODO Declare inputDim as a pointer and modify it
  // within inference function
  //Debug_PrintArray(input, 1, 7);
  // TODO Free all resources

  return 0;
}

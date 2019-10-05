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
  Onnx__ModelProto *model = openOnnxFile("../models/digits.onnx");
  if (model == NULL)
  {
    perror("Error when opening the onnx file\n");
    exit(-1);
  }

  /*This is model specific for digits.onnx and should be generalized. In this
  case the input dimensions are 1x64, so first dimension 1 is ignored*/
  //model->graph->input[0]->type->tensor_type->shape->n_dim
  //model->graph->input[0]->type->tensor_type->shape->dim[0]->dim_value
  //model->graph->input[0]->type->tensor_type->shape->dim[1]->dim_value
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

  return 0;
}

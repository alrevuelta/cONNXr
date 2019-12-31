#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pb/onnx.pb-c.h"
#include "embeddedml_debug.h"
#include "embeddedml_inference.h"
#include "embeddedml_utils.h"

int main(int argc, char **argv){

  /* TODO: Number of inputs is hardcoded to 1. The CLI is used with
  two parameters:
    - ONNX model to use
    - .pb input to feed (This has to be modified to support an arbitrary number)
  Example: connxr model.onnx input.pb
  */
  if (argc == 3){
    printf("Loading model %s...", argv[1]);
    Onnx__ModelProto *model = openOnnxFile(argv[1]);
    if (model != NULL){printf("ok!\n");}

    printf("Loading input %s...", argv[2]);
    Onnx__TensorProto *inp0set0 = openTensorProtoFile(argv[2]);
    if (inp0set0 != NULL){printf("ok!\n");}

    //Debug_PrintModelInformation(model);
    //debug_prettyprint_model(model);
    convertRawDataOfTensorProto(inp0set0);

    // TODO Dirty trick. I expected the input name to be included in the
    // input_0, but apparently it is not. Dont know if memory for the name
    // is allocated... but it doesnt crash
    inp0set0->name = "Input3"; // TODO hardcoded for MNIST
    TRACE_LEVEL0("%s\n\n", inp0set0->name);

    Onnx__TensorProto *inputs[] = { inp0set0 };
    clock_t start, end;
    double cpu_time_used;

    printf("Running inference on %s model...", model->graph->name);
    start = clock();
    Onnx__TensorProto **output = inference(model, inputs, 1);
    end = clock();
    printf("ok!\n"); // TODO Print nok if it fails

    // TODO Is CLOCKS_PER_SEC ok to use?
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Predicted in %f cycles or %f seconds\n", (double) (end - start), cpu_time_used);

    /* 11 is hardcoded, which is Plus214_Output_0 */
    for (int i = 0; i < output[11]->n_float_data; i++){
      printf("n_float_data[%d] = %f\n", i, output[11]->float_data[i]);
    }
  }else{
    printf("Wrong inputs\n");
  }
}

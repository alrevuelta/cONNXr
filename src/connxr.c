#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pb/onnx.pb-c.h"
#include "trace.h"
#include "inference.h"
#include "utils.h"

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

    printf("values = %d\n", inp0set0->data_type);

    inp0set0->name = model->graph->input[0]->name;
    TRACE_LEVEL0("%s\n", inp0set0->name);

    Onnx__TensorProto *inputs[] = { inp0set0 };
    clock_t start, end;
    double cpu_time_used;

    struct operator__context** all_op_context = resolve_check_get_input_and_attr(
                                                      model,
                                                      inputs,
                                                      1);

    printf("Running inference on %s model...\n", model->graph->name);
    start = clock();
    inference(all_op_context,
              model->graph->n_node);
    end = clock();
    printf("ok!\n"); // TODO Print nok if it fails

    // TODO Is CLOCKS_PER_SEC ok to use?
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Predicted in %f cycles or %f seconds\n", (double) (end - start), cpu_time_used);

    /* Last calculated output is 11 for mnist */
    //Onnx__TensorProto *mnist_output = *lazy_outputs_mapping_tensors[11];
    struct operator__onnx__add__context *out = (void *)(all_op_context[11]);
    Onnx__TensorProto *mnist_output = out->out->C;

    for (int i = 0; i < mnist_output->n_float_data; i++){
      printf("n_float_data[%d] = %f\n", i, mnist_output->float_data[i]);
    }
  }else{
    printf("Wrong inputs\n");
  }
}

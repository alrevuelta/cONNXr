#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "onnx.pb-c.h"
#include "tracing.h"
#include "inference.h"
#include "utils.h"

int main(int argc, char **argv){

  /* TODO: Number of inputs is hardcoded to 1. The CLI is used with
  two parameters:
    - ONNX model to use
    - .pb input to feed (This has to be modified to support an arbitrary number)
  Example: connxr model.onnx input.pb
  */
  if (argc <  3){
    fprintf(stderr, "not enough arguments! %s model.onnx input.pb [--dump-file]\n", argv[0]);
    return 1;
  }
  printf("Loading model %s...", argv[1]);
  Onnx__ModelProto *model = openOnnxFile(argv[1]);
  if (model != NULL){printf("ok!\n");}
  TRACE_MODEL(2, true, model);

  printf("Loading input %s...", argv[2]);
  Onnx__TensorProto *inp0set0 = openTensorProtoFile(argv[2]);
  if (inp0set0 != NULL){printf("ok!\n");}
  TRACE_TENSOR(2, true, inp0set0);

  //Debug_PrintModelInformation(model);
  //debug_prettyprint_model(model);
  convertRawDataOfTensorProto(inp0set0);

  printf("values = %" PRId32 "\n", inp0set0->data_type);

  inp0set0->name = model->graph->input[0]->name;

  Onnx__TensorProto *inputs[] = { inp0set0 };
  clock_t start, end;
  double cpu_time_used;

  printf("Resolving model...\n");
  resolve(model, inputs, 1);
  printf("Running inference on %s model...\n", model->graph->name);
  start = clock();
  inference(model, inputs, 1);
  end = clock();
  printf("finished!\n");

  // TODO Is CLOCKS_PER_SEC ok to use?
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Predicted in %f cycles or %f seconds\n", (double) (end - start), cpu_time_used);

  /* Print the last output which should be the model output */
  for (int i = 0; i < all_context[_populatedIdx].outputs[0]->n_float_data; i++){
    //printf("n_float_data[%d] = %f\n", i, all_context[_populatedIdx].outputs[0]->float_data[i]);
  }

  if ((argc == 4) && !strcmp(argv[3], "--dump-file")){
    printf("Writing dump file with intermediate outputs\n");
    //int max_print = 10;
    FILE *fp = fopen("dump.txt", "w+");
    for (int i = 0; i < _populatedIdx + 1; i++){
      fprintf(fp, "name=%s\n", all_context[i].outputs[0]->name);
      fprintf(fp, "shape=");
      for (int dim_index = 0; dim_index < all_context[i].outputs[0]->n_dims; dim_index++){
        fprintf(fp, "%" PRId64 ",", all_context[i].outputs[0]->dims[dim_index]);
      }
      fprintf(fp, "\n");
      //int float_to_print = all_context[i].outputs[0]->n_float_data > max_print ? max_print : all_context[i].outputs[0]->n_float_data;
      fprintf(fp, "tensor=");
      /* TODO: Just implemented for float */
      for (int data_index = 0; data_index < all_context[i].outputs[0]->n_float_data; data_index++){
        fprintf(fp, "%f,", all_context[i].outputs[0]->float_data[data_index]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tracing.h"
#include "utils.h"
#include "operators/ai.onnx/Constant/12/operator__ai_onnx__constant__12.h"

/* TODO We don't really need one function per type. So this function
is not for float, it covers everything */
operator_status
operator__ai_onnx__constant__12(
    node_context *ctx
)
{
  TRACE_ENTRY(1);

  if (ctx->onnx_node->n_attribute != 1){
    printf("Error, the number of attributes should be 1\n");
    exit(-1);
  }

  /* There are many attributes but only one is non NULL, hence only one
  is present*/
  Onnx__AttributeProto *value = ctx->onnx_node->attribute[0];

  TRACE_ATTRIBUTE(2, true, value);

  if (!strcmp(value->name, "sparse_value")){
    /*TODO*/
  }else if (!strcmp(value->name, "value")){

    /* Attention. Don't do this. Leaving it here so I don't waste again
    hours debugging this. Don't modify the memory address that the
    output is pointing. Otherwise the next operator in the node won't
    be able to find it. Copy it instead*/
    //ctx->outputs[0] = value->t;

    char *name = ctx->outputs[0]->name;
    memcpy(ctx->outputs[0], value->t, sizeof(Onnx__TensorProto));
    ctx->outputs[0]->name = name;
    convertRawDataOfTensorProto(ctx->outputs[0]);

  }else if (!strcmp(value->name, "value_float")){
  }else if (!strcmp(value->name, "value_floats")){
  }else if (!strcmp(value->name, "value_int")){
    printf("entering value_int\n");
  }else if (!strcmp(value->name, "value_ints")){
  }else if (!strcmp(value->name, "value_string")){
  }else if (!strcmp(value->name, "value_strings")){
  }

  TRACE_TENSOR(2, true, ctx->outputs[0]);
  TRACE_EXIT(1);

  return 0;
}

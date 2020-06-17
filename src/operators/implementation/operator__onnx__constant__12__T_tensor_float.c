#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trace.h"
#include "utils.h"

/* TODO Not sure if we need one function per type. Maybe some macros ? */
operator_status operator__onnx__constant__12__T_tensor_float(
    node_context *ctx
)
{

  Onnx__AttributeProto *value = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "value");
  Onnx__AttributeProto *value_float = searchAttributeNyName(ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "value_float");

  printf("name value = %s\n", value->name);
  printf("n_tensors %zu\n", value->n_tensors);
  printf("type %u\n", value->type);

  /* Handling only one particular case */
  if (value->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR){
    printf("n_dims %zu", value->t->n_dims);
    printf("dims[0] %lld", value->t->dims[0]);

    // No mem is allocated, just point the output to the already stored
    // tensor in the attribute
    ctx->outputs[0] = value->t;
  }/* More cases to handle, float, ints, string,...*/

  return 0;
}

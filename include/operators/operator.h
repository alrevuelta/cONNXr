#ifndef OPERATOR_H
#define OPERATOR_H

typedef struct operator_context operator_context;
typedef enum operator_status    operator_status;
typedef operator_status (*operator_executer)(operator_context *ctx);
typedef operator_executer (*operator_resolver)(operator_context *ctx);

#include "onnx.pb-c.h"
#include <errno.h>
#include "list.h"
#include "operators/operator_info.h"
#include <stdbool.h>

enum operator_status {
  OP_OK     = 0,
  OP_ENOSYS = ENOSYS, // Function not implemented
  OP_ENOMEM = ENOMEM, // Out of memory
  OP_EINVAL = EINVAL, // Invalid argument
  OP_EDOM   = EDOM,   // Math argument out of domain of func
  OP_ERANGE = ERANGE  // Math result not representable
};

struct operator_info;
struct operator_context
{
    Onnx__NodeProto       *node;
    Onnx__TensorProto    **input;
    Onnx__TensorProto    **output;
    Onnx__AttributeProto **attribute;
    operator_info         *info;
    operator_executer      executor;
};


operator_context*
operator_context_create(operator_info    *info,
                        Onnx__NodeProto  *node);

bool
operator_context_link(Onnx__ModelProto *model,
                      operator_context *ctx);

bool
operator_context_linkAll(Onnx__ModelProto *model);


bool
operator_context_createAll(Onnx__ModelProto *model);


#endif
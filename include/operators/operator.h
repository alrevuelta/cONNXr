#ifndef OPERATOR_H
#define OPERATOR_H

#include "onnx.pb-c.h"
#include <errno.h>

typedef struct operator_context operator_context;
typedef struct operator_tensor           operator_tensor;
typedef struct operator_context_tensor     operator_context_input;
typedef struct operator_context_tensor     operator_context_output;
typedef struct operator_context_attribute  operator_context_attribute;
typedef enum operator_status    operator_status;
typedef operator_status (*operator_executer)(void *ctx);
typedef operator_executer (*operator_resolver)(void *ctx);

enum operator_status {
  OP_OK     = 0,
  OP_ENOSYS = ENOSYS, // Function not implemented
  OP_ENOMEM = ENOMEM, // Out of memory
  OP_EINVAL = EINVAL, // Invalid argument
  OP_EDOM   = EDOM,   // Math argument out of domain of func
  OP_ERANGE = ERANGE  // Math result not representable
};

struct operator_tensor
{
    Onnx__TensorProto *tensor;
    operator_context  *origin;
    char              *name;
};

struct operator_context_tensor
{
    size_t  length;
    operator_tensor *tensor[];
};

struct operator_context_attribute
{
    size_t                length;
    Onnx__AttributeProto *attribute[];
};

struct operator_context
{
    operator_context_input      *input;
    operator_context_output     *output;
    operator_context_attribute  *attribute;
    operator_executer operator;
};

#endif
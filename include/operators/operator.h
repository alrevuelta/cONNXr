#ifndef OPERATOR_H
#define OPERATOR_H

#include "onnx.pb-c.h"
#include <errno.h>

typedef struct operator_context operator_context;
typedef struct tensor           tensor;
typedef struct array_tensor     array_input;
typedef struct array_tensor     array_output;
typedef struct array_attribute  array_attribute;
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

struct tensor
{
    Onnx__TensorProto *tensor;
    operator_context  *origin;
    char              *name;
};

struct array_tensor
{
    size_t  length;
    tensor *tensor[];
};

struct array_attribute
{
    size_t                length;
    Onnx__AttributeProto *attribute[];
};

struct operator_context
{
    array_input      *input;
    array_output     *output;
    array_attribute  *attribute;
    operator_executer operator;
};

#endif
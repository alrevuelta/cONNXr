#ifndef OPERATOR_H
#define OPERATOR_H

#include "onnx.pb-c.h"
#include <errno.h>

// TODO Remove unused code
typedef enum operator_status operator_status;
typedef struct node_context  node_context;
typedef operator_status (*operator_executer)(node_context *ctx);
typedef operator_executer (*operator_resolver)(node_context *ctx);



// TODO Move this to a file named operator_interface
struct node_context {
  Onnx__NodeProto     *onnx_node;
  Onnx__TensorProto  **inputs;
  Onnx__TensorProto  **outputs;
  operator_executer resolved_op;
  //int (*resolved_op)(node_context *ctx);
};

enum operator_status {
  OP_OK     = 0,
  OP_ENOSYS = ENOSYS, // Function not implemented
  OP_ENOMEM = ENOMEM, // Out of memory
  OP_EINVAL = EINVAL, // Invalid argument
  OP_EDOM   = EDOM,   // Math argument out of domain of func
  OP_ERANGE = ERANGE  // Math result not representable
};

#endif

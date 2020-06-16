#include <stdio.h>

#include "operators/operator_stub.h"

operator_status operator_stub(
    node_context *ctx
) {
    fprintf(stderr, "operator not implemented!\n");
    return OP_ENOSYS;
}

#include <stdio.h>

#include "operators/operator_stub.h"

operator_info operator_stub_info = {
    .name         = "stub",
    .resolver     = &operator_stub_resolver,
    .range_input  = {0, SIZE_MAX},
    .range_output = {0, SIZE_MAX},
    .n_attribute  = 0,
    .attribute    = NULL,
    .n_input      = 0,
    .input        = NULL,
    .n_output     = 0,
    .output       = NULL,
    .n_constraint = 0,
    .constraint   = NULL,
};

operator_executer operator_stub_resolver(operator_context *ctx)
{
    return &operator_stub;
}

operator_status operator_stub(operator_context *ctx)
{
    fprintf(stderr, "operator not implemented!\n");
    return OP_ENOSYS;
}
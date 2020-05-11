#ifndef OPERATOR_STUB_H
#define OPERATOR_STUB_H

#include "operators/operator.h"

extern operator_info operator_stub_info;

operator_status operator_stub(operator_context *ctx);

operator_executer operator_stub_resolver(operator_context *ctx);

#endif
#ifndef OPERATOR_CHECK_H
#define OPERATOR_CHECK_H

#include "operators/operator.h"

bool
operator_check(operator_context *ctx);

bool
operator_check_range(operator_context *ctx);

bool
operator_check_attributes(operator_context *ctx);

bool
operator_check_tensors(operator_context *ctx);

bool
operator_check_constraint(operator_context *ctx);

#endif
#ifndef OPERATOR_CHECK_H
#define OPERATOR_CHECK_H

#include "operators/operator.h"
#include "operators/operator_info.h"
#include "stdbool.h"

bool
operator_check(node_context *ctx, operator_info *info);

bool
operator_check_range(node_context *ctx, operator_info *info);

bool
operator_check_attributes(node_context *ctx, operator_info *info);

bool
operator_check_tensors(node_context *ctx, operator_info *info);

bool
operator_check_constraint(node_context *ctx, operator_info *info);

#endif
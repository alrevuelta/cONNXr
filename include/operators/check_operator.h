#ifndef CHECK_OPERATOR_H
#define CHECK_OPERATOR_H

#include "onnx.pb-c.h"
#include <stdbool.h>

typedef struct check_operator_condition_attribute
{
  bool skip;
  char* name;
  bool optional;
  uint32_t type;
} check_operator_condition_attribute;

typedef struct check_operator_condition_tensor
{
  bool skip;
  char* name;
  bool optional;
  size_t n_types;
  uint32_t *types;
} check_operator_condition_tensor;

typedef struct check_operator_condition_constraint
{
  bool skip;
  char* name;
  bool optional;
} check_operator_condition_constraint;

typedef struct check_operator_condition_range
{
  char *name;
  size_t min;
  size_t max;
} check_operator_condition_range;

bool
check_operator_range(char* prefix,
                     check_operator_condition_range* condition,
                     size_t number);

bool
check_operator_attributes(char* prefix,
                          size_t n_conditions,
                          check_operator_condition_attribute* conditions,
                          Onnx__AttributeProto** attributes);

bool
check_operator_tensors(char* prefix,
                       size_t n_conditions,
                       check_operator_condition_tensor* conditions,
                       Onnx__TensorProto** tensors);

bool
check_operator_constraint(
  char* prefix,
  size_t n_conditions_input,
  check_operator_condition_constraint* conditions_input,
  Onnx__TensorProto** inputs,
  size_t n_conditions_output,
  check_operator_condition_constraint* conditions_output,
  Onnx__TensorProto** outputs);

#endif
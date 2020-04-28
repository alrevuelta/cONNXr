#ifndef OPERATOR_CHECK_H
#define OPERATOR_CHECK_H

#include "onnx.pb-c.h"
#include <stdbool.h>

typedef struct operator_check_condition_attribute
{
  bool skip;
  char* name;
  bool optional;
  uint32_t type;
} operator_check_condition_attribute;

typedef struct operator_check_condition_tensor
{
  bool skip;
  char* name;
  bool optional;
  size_t n_types;
  uint32_t *types;
} operator_check_condition_tensor;

typedef struct operator_check_condition_constraint
{
  bool skip;
  char* name;
  bool optional;
} operator_check_condition_constraint;

typedef struct operator_check_condition_range
{
  char *name;
  size_t min;
  size_t max;
} operator_check_condition_range;

bool
operator_check_range(char* prefix,
                     operator_check_condition_range* condition,
                     size_t number);

bool
operator_check_attributes(char* prefix,
                          size_t n_conditions,
                          operator_check_condition_attribute* conditions,
                          Onnx__AttributeProto** attributes);

bool
operator_check_tensors(char* prefix,
                       size_t n_conditions,
                       operator_check_condition_tensor* conditions,
                       Onnx__TensorProto** tensors);

bool
operator_check_constraint(
  char* prefix,
  size_t n_conditions_input,
  operator_check_condition_constraint* conditions_input,
  Onnx__TensorProto** inputs,
  size_t n_conditions_output,
  operator_check_condition_constraint* conditions_output,
  Onnx__TensorProto** outputs);

#endif
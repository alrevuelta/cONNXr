#include "operators/check_operator.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

const char*
attributeTypeString(uint32_t type)
{
  switch (type) {
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED:
      return "UNDEFINED";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
      return "FLOAT";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
      return "INT";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
      return "STRING";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
      return "TENSOR";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH:
      return "GRAPH";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
      return "SPARSE_TENSOR";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
      return "FLOATS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
      return "INTS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
      return "STRINGS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS:
      return "TENSORS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS:
      return "GRAPHS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS:
      return "SPARSE_TENSORS";
    default:
      return NULL;
  }
}

const char*
tensorTypeString(uint32_t type)
{
  switch (type) {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      return "UNDEFINED";
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      return "FLOAT";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      return "UINT8";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      return "INT8";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      return "UINT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      return "INT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      return "INT32";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      return "INT64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      return "STRING";
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      return "BOOL";
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      return "FLOAT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      return "DOUBLE";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      return "UINT32";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      return "UINT64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      return "COMPLEX64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      return "COMPLEX128";
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      return "BFLOAT16";
    default:
      return NULL;
  }
}

bool
check_operator_range(char* prefix,
                     check_operator_condition_range* condition,
                     size_t number)
{
  if (number < condition->min) {
    fprintf(stderr,
            "%s: %s number is too small! "
            "expected at least %" PRId64 ", but got %" PRId64 "\n",
            prefix,
            condition->name,
            condition->min,
            number);
    return false;
  }
  if (number > condition->max) {
    fprintf(stderr,
            "%s: %s number is too large! "
            "expected at most %" PRId64 ", but got %" PRId64 "\n",
            prefix,
            condition->name,
            condition->max,
            number);
    return false;
  }
  return true;
}

bool
check_operator_attributes(char* prefix,
                          size_t n_conditions,
                          check_operator_condition_attribute* conditions,
                          Onnx__AttributeProto** attributes)
{
  for (size_t i_cond = 0; i_cond < n_conditions; i_cond++) {
    check_operator_condition_attribute* cond = &conditions[i_cond];
    Onnx__AttributeProto* attr = attributes[i_cond];
    if (cond->skip) {
      continue;
    }
    if (!attr) {
      if (conditions[i_cond].optional) {
        continue;
      }
      fprintf(stderr,
              "%s: missing non-optional attribute '%s' at pos %" PRIu64 "!\n",
              prefix,
              conditions[i_cond].name,
              i_cond);
      return false;
    }
    if (strcmp(conditions[i_cond].name, attr->name) != 0) {
      fprintf(stderr,
              "%s: attribute '%s' at pos %" PRIu64 " has wrong name '%s'\n",
              prefix,
              conditions[i_cond].name,
              i_cond,
              attr->name);
      return false;
    }
    if (conditions[i_cond].type != attr->type) {
      fprintf(stderr,
              "%s: attribute '%s' at pos %" PRIu64 " has wrong type! "
              "got '%s', but expected '%s'\n",
              prefix,
              conditions[i_cond].name,
              i_cond,
              attributeTypeString(conditions[i_cond].type),
              attributeTypeString(attr->type));
      return false;
    }
  }
  return true;
}

bool
check_operator_tensors(char* prefix,
                       size_t n_conditions,
                       check_operator_condition_tensor* conditions,
                       Onnx__TensorProto** tensors)
{
  for (size_t i_cond = 0; i_cond < n_conditions; i_cond++) {
    check_operator_condition_tensor* cond = &conditions[i_cond];
    Onnx__TensorProto* tensor = tensors[i_cond];
    if (cond->skip) {
      continue;
    }
    if (!tensor) {
      if (cond->optional) {
        continue;
      }
      fprintf(stderr,
              "%s: did not found non-optional tensor '%s' at pos %" PRIu64 "!\n",
              prefix,
              cond->name,
              i_cond);
      return false;
    }
    bool validType = false;
    for (size_t i_type = 0; i_type < cond->n_types; i_type++) {
      if (cond->types[i_type] == tensor->data_type) {
        validType = true;
        break;
      }
    }
    if (!validType) {
      fprintf(stderr,
              "%s: tensor '%s' ('%s') at pos %" PRIu64 " has wrong type! "
              "got '%s', but expected one of ",
              prefix,
              cond->name,
              tensor->name,
              i_cond,
              tensorTypeString(tensor->data_type));
      for (size_t i_type = 0; i_type < cond->n_types; i_type++) {
        uint32_t type = cond->types[i_type];
        fprintf(stderr, "'%s'", tensorTypeString(type));
        if (i_type + 1 != cond->n_types) {
          fprintf(stderr, ", ");
        }
      }
      fprintf(stderr, "\n");
      return false;
    }
  }
  return true;
}

bool
check_operator_constraint(
  char* prefix,
  size_t n_conditions_input,
  check_operator_condition_constraint* conditions_input,
  Onnx__TensorProto** inputs,
  size_t n_conditions_output,
  check_operator_condition_constraint* conditions_output,
  Onnx__TensorProto** outputs)
{
  char* kind_input = "input";
  char* kind_output = "output";

  bool ref_valid = false;
  uint32_t ref_type = 0;
  char* ref_name_model = NULL;
  char* ref_name_cond = NULL;
  char* ref_kind = NULL;

  for (size_t i_cond = 0; i_cond < n_conditions_input; i_cond++) {
    check_operator_condition_constraint* cond = &conditions_input[i_cond];
    Onnx__TensorProto* input = inputs[i_cond];
    if (cond->skip) {
      continue;
    }
    if (!ref_valid) {
      ref_type = input->data_type;
      ref_name_model = input->name;
      ref_name_cond = cond->name;
      ref_kind = kind_input;
      continue;
    }
    if (input->data_type != ref_type) {
      fprintf(stderr,
              "%s: shared constraint violation! "
              "%s '%s' ('%s') has type '%s' and "
              "%s '%s' ('%s') has type '%s'\n",
              prefix,
              ref_kind,
              ref_name_cond,
              ref_name_model,
              tensorTypeString(ref_type),
              kind_input,
              cond->name,
              input->name,
              tensorTypeString(input->data_type));
      return false;
    }
  }
  for (size_t i_cond = 0; i_cond < n_conditions_output; i_cond++) {
    check_operator_condition_constraint* cond = &conditions_output[i_cond];
    Onnx__TensorProto* output = outputs[i_cond];
    if (cond->skip) {
      continue;
    }
    if (!ref_valid) {
      ref_type = output->data_type;
      ref_name_model = output->name;
      ref_name_cond = cond->name;
      ref_kind = kind_input;
      continue;
    }
    if (output->data_type != ref_type) {
      fprintf(stderr,
              "%s: shared constraint violation! "
              "%s '%s' ('%s') has type '%s' and "
              "%s '%s' ('%s') has type '%s'\n",
              prefix,
              ref_kind,
              ref_name_cond,
              ref_name_model,
              tensorTypeString(ref_type),
              kind_output,
              cond->name,
              output->name,
              tensorTypeString(output->data_type));
      return false;
    }
  }
  return true;
}

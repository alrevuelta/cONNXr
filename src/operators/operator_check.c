#include "operators/operator_check.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static bool
operator_check_range_(char                *operator,
                      char                *name,
                      operator_info_range *info,
                      size_t               number)
{
  if (number < info->min) {
    fprintf(stderr,
            "%s:%s input number is too small! "
            "expected at least %zu, but got %zu\n",
            operator,
            name,
            info->min,
            number);
    return false;
  }
  if (number > info->max) {
    fprintf(stderr,
            "%s:%s input number is too large! "
            "expected at most %zu, but got %zu\n",
            operator,
            name,
            info->max,
            number);
    return false;
  }
  return true;
}

bool
operator_check_range(operator_context *ctx)
{
  char *name = (ctx->node->name) ? ctx->node->name : "";
  bool valid = true;
  valid &= operator_check_range_(ctx->info->name,
                                 name,
                                 &ctx->info->range_input,
                                 ctx->node->n_input);
  valid &= operator_check_range_(ctx->info->name,
                                 name,
                                 &ctx->info->range_output,
                                 ctx->node->n_output);
  return valid;
}

bool
operator_check_attributes(operator_context *ctx)
{
  char *name = (ctx->node->name) ? ctx->node->name : "";
  for (size_t i_attr = 0; i_attr < ctx->info->n_attribute; i_attr++) {
    operator_info_attribute* cond = &ctx->info->attribute[i_attr];
    Onnx__AttributeProto* cattr = ctx->attribute[i_attr];
    if (!cattr) {
      if (cond->optional) {
        continue;
      }
      fprintf(stderr,
              "%s:%s missing non-optional attribute '%s' at pos %zu!\n",
              ctx->info->name,
              name,
              cond->name,
              i_attr);
      return false;
    }
    if (strcmp(cond->name, cattr->name) != 0) {
      fprintf(stderr,
              "%s:%s attribute '%s' at pos %zu has wrong name '%s'\n",
              ctx->info->name,
              name,
              cond->name,
              i_attr,
              cattr->name);
      return false;
    }
    if (cond->type != cattr->type) {
      fprintf(stderr,
              "%s:%s attribute '%s' at pos %zu has wrong type! "
              "got '%s', but expected '%s'\n",
              ctx->info->name,
              name,
              cond->name,
              i_attr,
              operator_info_attributeType2str(cond->type),
              operator_info_attributeType2str(cattr->type));
      return false;
    }
  }
  return true;
}

static bool
operator_check_tensor_type(char                  *operator,
                           char                  *name,
                           Onnx__TensorProto     *tensor,
                           size_t                 pos,
                           operator_info_tensor  *info)
{
  bool validType = false;
  for (size_t i_type = 0; i_type < info->n_types; i_type++) {
    if (info->types[i_type] == tensor->data_type) {
      validType = true;
      break;
    }
  }
  if (!validType) {
    fprintf(stderr,
            "%s:%s tensor '%s' ('%s') at pos %zu has wrong type! "
            "got '%s', but expected one of ",
            operator,
            name,
            info->name,
            tensor->name,
            pos,
            operator_info_tensorType2str(tensor->data_type));
    for (size_t i_type = 0; i_type < info->n_types; i_type++) {
      uint32_t type = info->types[i_type];
      fprintf(stderr, "'%s'", operator_info_tensorType2str(type));
      if (i_type + 1 != info->n_types) {
        fprintf(stderr, ", ");
      }
    }
    fprintf(stderr, "\n");
    return false;
  }
  return true;
}

static bool
operator_check_tensors_(char                  *operator,
                        char                  *name,
                        size_t                 n_tensor,
                        Onnx__TensorProto    **tensor,
                        size_t                 n_info,
                        operator_info_tensor  *info)
{
  for (size_t i_tensor = 0; i_tensor < n_info; i_tensor++) {
    operator_info_tensor* cond = &info[i_tensor];
    Onnx__TensorProto* ctensor = tensor[i_tensor];
    if (!ctensor) {
      if (cond->optional) {
        continue;
      }
      fprintf(stderr,
              "%s:%s did not found non-optional tensor '%s' at pos %zu!\n",
              operator,
              name,
              cond->name,
              i_tensor);
      return false;
    }
    bool valid = operator_check_tensor_type(operator,
                                            name,
                                            ctensor,
                                            i_tensor,
                                            cond);
    if (!valid) {
      return false;
    }
  }
  /* check if we have variadic tensors */
  if (n_tensor < n_info || n_info == 0) {
    return true;
  }
  operator_info_tensor* cond = &info[n_info-1];
  uint32_t type = 0;
  if (!cond->homogeneous) {
    type = tensor[n_info-1]->data_type;
  }

  for (size_t i_tensor = n_info; i_tensor < n_tensor; i_tensor++) {
    Onnx__TensorProto* ctensor = tensor[i_tensor];
    if (type && type != ctensor->data_type) {
      fprintf(stderr,
              "%s:%s tensor '%s' ('%s') at pos %zu has wrong type! "
              "got '%s', but expected homogeneous one '%s'\n",
              operator,
              name,
              cond->name,
              ctensor->name,
              i_tensor,
              operator_info_tensorType2str(ctensor->data_type),
              operator_info_tensorType2str(type));
      return false;
    } else {
      bool valid = operator_check_tensor_type(operator,
                                              name,
                                              ctensor,
                                              i_tensor,
                                              cond);
      if (!valid) {
        return false;
      }
    }
  }

  return true;
}

bool
operator_check_tensors(operator_context *ctx)
{
  char *name = (ctx->node->name) ? ctx->node->name : "";
  bool valid = true;
  valid &= operator_check_tensors_(ctx->info->name,
                                   name,
                                   ctx->node->n_input,
                                   ctx->input,
                                   ctx->info->n_input,
                                   ctx->info->input);
  valid &= operator_check_tensors_(ctx->info->name,
                                   name,
                                   ctx->node->n_output,
                                   ctx->output,
                                   ctx->info->n_output,
                                   ctx->info->output);
  return valid;
}

bool
operator_check_constraint(operator_context *ctx)
{
  char *name = (ctx->node->name) ? ctx->node->name : "";

  for (size_t i_cons = 0; i_cons < ctx->info->n_constraint; i_cons++) {
    operator_info_constraint *constraint = &ctx->info->constraint[i_cons];
    Onnx__TensorProto    *ref      = NULL;
    char                 *ref_type = NULL;
    operator_info_tensor *ref_cond = NULL;
    for (size_t i_tensor = 0; i_tensor < ctx->info->n_input; i_tensor++) {
      Onnx__TensorProto    *tensor = ctx->input[i_tensor];
      operator_info_tensor *cond   = &ctx->info->input[i_tensor];
      if (!tensor) {
        continue;
      }
      if (strcmp(cond->constraint,constraint->name)==0) {
        if (!ref) {
          ref = tensor;
          ref_type = "input";
          ref_cond = cond;
          continue;
        }
        if (tensor->data_type != ref->data_type) {
          fprintf(stderr,
                  "%s:%s shared constraint violation! "
                  "%s '%s' ('%s') has type '%s' and "
                  "%s '%s' ('%s') has type '%s'\n",
                  ctx->info->name,
                  name,
                  ref_type,
                  ref_cond->name,
                  ref->name,
                  operator_info_tensorType2str(ref->data_type),
                  "input",
                  cond->name,
                  tensor->name,
                  operator_info_tensorType2str(tensor->data_type));
          return false;
        }
      }
    }
    for (size_t i_tensor = 0; i_tensor < ctx->info->n_output; i_tensor++) {
      Onnx__TensorProto    *tensor = ctx->output[i_tensor];
      operator_info_tensor *cond = &ctx->info->output[i_tensor];
      if (!tensor) {
        continue;
      }
      if (strcmp(cond->constraint,constraint->name)==0) {
        if (!ref) {
          ref = tensor;
          ref_type = "output";
          ref_cond = cond;
          continue;
        }
        if (tensor->data_type != ref->data_type) {
          fprintf(stderr,
                  "%s:%s shared constraint violation! "
                  "%s '%s' ('%s') has type '%s' and "
                  "%s '%s' ('%s') has type '%s'\n",
                  ctx->info->name,
                  name,
                  ref_type,
                  ref_cond->name,
                  ref->name,
                  operator_info_tensorType2str(ref->data_type),
                  "output",
                  cond->name,
                  tensor->name,
                  operator_info_tensorType2str(tensor->data_type));
          return false;
        }
      }
    }
  }
  return true;
}

bool
operator_check(operator_context *ctx)
{
  bool valid = true;
  valid &= operator_check_range(ctx);
  valid &= operator_check_attributes(ctx);
  valid &= operator_check_tensors(ctx);
  valid &= operator_check_constraint(ctx);
  return valid;
}
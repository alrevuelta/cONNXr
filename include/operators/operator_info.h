#ifndef OPERATOR_INFO_H
#define OPERATOR_INFO_H

typedef struct operator_info_attribute  operator_info_attribute;
typedef struct operator_info_tensor     operator_info_tensor;
typedef struct operator_info_constraint operator_info_constraint;
typedef struct operator_info_range      operator_info_range;
typedef struct operator_info            operator_info;

#include "onnx.pb-c.h"
#include "operators/operator.h"
#include <stdbool.h>

struct operator_info_attribute
{
  char    *name;
  bool     optional;
  uint32_t type;
};

struct operator_info_tensor
{
  char     *name;
  bool      optional;
  bool      variadic;
  bool      homogeneous;
  char     *constraint;
  size_t    n_types;
  uint32_t *types;
};

struct operator_info_constraint
{
  char *name;
};

struct operator_info_range
{
  size_t min;
  size_t max;
};

struct operator_info
{
  char                     *name;
  operator_info_range       range_input;
  operator_info_range       range_output;
  size_t                    n_attribute;
  operator_info_attribute  *attribute;
  size_t                    n_input;
  operator_info_tensor     *input;
  size_t                    n_output;
  operator_info_tensor     *output;
  size_t                    n_constraint;
  operator_info_constraint *constraint;
};

operator_info*
operator_info_find(size_t  version,
                   char   *domain,
                   char   *name);

char*
operator_info_attributeType2str(uint32_t type);

char*
operator_info_tensorType2str(uint32_t type);

#endif
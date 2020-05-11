#ifndef OPERATOR_SETS_H
#define OPERATOR_SETS_H

#include "operators/operator_info.h"

typedef struct operator_set       operator_set;
typedef struct operator_sets      operator_sets;

struct operator_set
{
    size_t         version;
    char          *domain;
    size_t         length;
    operator_info *entries[];
};

struct operator_sets
{
    size_t        length;
    operator_set *sets[];
};

extern operator_sets onnx_operator_sets;

#endif

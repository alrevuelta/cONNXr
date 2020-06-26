#ifndef OPERATOR_SETS_H
#define OPERATOR_SETS_H

typedef struct operator_set       operator_set;
typedef struct operator_sets      operator_sets;
typedef struct operator_set_entry operator_set_entry;

#include "operators/operator_info.h"
#include "operators/operator.h"

struct operator_set_entry {
    char             *name;
    operator_resolver resolver;
    operator_info    *info;
};

struct operator_set
{
    size_t              version;
    char               *domain;
    size_t              length;
    operator_set_entry *entries[];
};

struct operator_sets
{
    size_t        length;
    operator_set *sets[];
};

extern operator_sets onnx_operator_sets;

#endif

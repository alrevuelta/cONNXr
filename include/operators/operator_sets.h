#ifndef OPERATOR_SETS_H
#define OPERATOR_SETS_H

#include "operator.h"
#include <stddef.h>

typedef struct operator_set_entry operator_set_entry;
typedef struct operator_set       operator_set;
typedef struct operator_sets      operator_sets;

struct operator_set_entry
{
    char             *name;
    operator_resolver resolver;
};

struct operator_set
{
    size_t              version;
    char               *domain;
    size_t              length;
    operator_set_entry  entries[];

};

struct operator_sets
{
    size_t        length;
    operator_set *sets[];
};

extern operator_sets all_operator_sets;

#endif

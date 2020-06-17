#ifndef OPERATOR_SETS_H
#define OPERATOR_SETS_H

#include "operator.h"
#include <stddef.h>
#include <string.h>
#include <stdio.h>

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

static __attribute__((unused))
operator_resolver find_operator_resolver(
    char *name,
    size_t version
) {
    for( size_t i_set = 0; i_set < all_operator_sets.length; i_set++ ) {
        operator_set *set = all_operator_sets.sets[i_set];
        if (set->version != version) {
            continue;
        }
        for (size_t i_entry = 0; i_entry < set->length; i_entry++) {
            operator_set_entry *entry = &set->entries[i_entry];
            if (strcmp(entry->name,name) == 0) {
                printf("Found opname:%s version:%zu\n", name, version);
                return entry->resolver;
            }
        }
    }
    printf("Resolver not found opname:%s version:%zu\n", name, version);
    // TODO Break here? Doesn't make sense to continue if the resolver is
    // not found
    return NULL;
}

#endif

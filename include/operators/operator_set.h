#ifndef OPERATOR_SET_H
#define OPERATOR_SET_H

#include "operator.h"
#include "tracing.h"
#include <stddef.h>
#include <string.h>

typedef struct operator_set_opdomain  operator_set_opdomain;
typedef struct operator_set_opname    operator_set_opname;
typedef struct operator_set_opversion operator_set_opversion;

//TODO clean up includes
#include "operators/operator_info.h"
#include "operators/operator.h"

struct operator_set_opversion
{
    size_t             version;
    operator_preparer  preparer;
    operator_info     *info;
};

struct operator_set_opname
{
    char                   *name;
    operator_set_opversion *opversions[];
};

struct operator_set_opdomain
{
    char                *name;
    operator_set_opname *opnames[];
};

extern operator_set_opdomain *operator_set[];

static __attribute__((unused))
operator_preparer
operator_set_find_preparer(
    char *name,
    size_t version
) {
    operator_set_opversion *tmp = NULL;
    for (operator_set_opdomain **opdomain = operator_set; *opdomain; opdomain++)
    {
        for (operator_set_opname **opname = (*opdomain)->opnames; *opname; opname++)
        {
            if (strcmp((*opname)->name,name) == 0) {
                for (operator_set_opversion **opversion = (*opname)->opversions; *opversion; opversion++)
                {
                    if ((*opversion)->version <= version) {
                        if (!tmp || (*opversion)->version >= tmp->version) {
                            tmp = *opversion;
                        }
                    }
                }
                if (tmp) {
                    TRACE(2, true, "Found operator '%s' version '%zu'", name, tmp->version);
                    return tmp->preparer;
                }
            }
        }
    }
    TRACE_FATAL(0, true, "No Operator not found  with name '%s' for opset '%zu'", name, tmp->version);
    return NULL;
}

static __attribute__((unused))
operator_info*
operator_set_find_info(
    char *name,
    size_t version)
{
    operator_set_opversion *tmp = NULL;
    for (operator_set_opdomain **opdomain = operator_set; *opdomain; opdomain++)
    {
        for (operator_set_opname **opname = (*opdomain)->opnames; *opname; opname++)
        {
            if (strcmp((*opname)->name, name) == 0)
            {
                for (operator_set_opversion **opversion = (*opname)->opversions; *opversion; opversion++)
                {
                    if ((*opversion)->version <= version)
                    {
                        if (!tmp || (*opversion)->version >= tmp->version)
                        {
                            tmp = *opversion;
                        }
                    }
                }
                if (tmp)
                {
                    TRACE(2, true, "Found operator '%s' version '%zu'", name, tmp->version);
                    return tmp->info;
                }
            }
        }
    }
    TRACE_FATAL(0, true, "No Operator not found  with name '%s' for opset '%zu'", name, tmp->version);
    return NULL;
}

#endif

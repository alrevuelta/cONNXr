#ifndef INDEX_H
#define INDEX_H

#include <stddef.h>
#include <stdint.h>
#include "tracing.h"

#define FOR_INDEX(INDEX) for( index_reset(INDEX); index_valid(INDEX); index_inc(INDEX) )
#define FOR_INDEX_SUB(INDEX,START) for( index_reset_sub(INDEX,START); index_valid_sub(INDEX,START); index_inc_sub(INDEX,START) )

#ifdef TRACE_LEVEL

#define TRACE_INDEX(LEVEL, COND, INDEX) \
if ((COND) && __TRACE_COND(LEVEL)) { \
    __VAR(LEVEL, "INDEX", (INDEX)->length, " %zu\n") \
    __VAR(LEVEL, "INDEX", (INDEX)->indices, "%p ") \
    __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                  (INDEX)->indices, \
                  , \
                  (INDEX)->length, \
                  "%" PRId64) \
    __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    __VAR(LEVEL, "INDEX", (INDEX)->limits, " %p ") \
    __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                  (INDEX)->limits, \
                  , \
                  (INDEX)->length, \
                  "%" PRId64) \
    __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    __VAR(LEVEL, "INDEX", (INDEX)->offsets, "%p ") \
    __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                  (INDEX)->offsets, \
                  , \
                  (INDEX)->length, \
                  "%" PRId64) \
    __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    __VAR(LEVEL, "INDEX", (INDEX)->offset, " %" PRId64 "\n") \
}

#else

#define TRACE_INDEX(LEVEL, COND, INDEX) ;

#endif


typedef struct index_ctx index_ctx;

struct index_ctx {
    size_t length;
    int64_t *limits;
    int64_t *indices;
    int64_t *offsets;
    int64_t offset;
};

static inline
void
index_init(index_ctx *index, size_t length, int64_t *limits, int64_t *indices, int64_t *offsets) {
    index->length  = length;
    index->limits  = limits;
    index->indices = indices;
    index->offsets = offsets;
    index->offset  = 0;
    for (int i = 0; i < index->length; i++) {
      index->indices[i] = 0;
    }
    index->offsets[index->length-1] = 1;
    for (int i = index->length-2; i >= 0; i--) {
      index->offsets[i] = index->limits[i+1] * index->offsets[i+1];
    }
}

static inline
void
index_reset_sub(index_ctx *index, int start) {
    for (int i = start; i < index->length; i++) {
      index->offset -= index->indices[i]*index->offsets[i];
      index->indices[i] = 0;
    }
}

static inline
void
index_reset(index_ctx *index) {
    return index_reset_sub(index,0);
}

static inline
bool
index_inc_sub(index_ctx *index, int stop) {
    index->offset++;
    index->indices[index->length-1]++;
    for (int i = index->length-1; i > stop; i--) {
        if (index->indices[i] < index->limits[i]) {
            return true;
        }
        index->indices[i] = 0;
        index->indices[i-1] += 1;
    }
    return false;
}

static inline
bool
index_inc(index_ctx *index) {
    return index_inc_sub(index,0);
}

static inline
bool
index_valid_sub(index_ctx *index, int start) {
    return index->indices[start] < index->limits[start];
}

static inline
bool
index_valid(index_ctx *index) {
    return index_valid_sub(index, 0);
}

static inline
void
index_set(index_ctx *index, size_t dim, int64_t value) {
    // TRACE(3,"old[%d]: %d",dim, index->indices[dim]);
    // TRACE(3,"new[%d]: %d",dim, value);
    int64_t diff = value - index->indices[dim];
    index->indices[dim] = value;
    // TRACE(3,"old offset: %d",index->offset);
    index->offset += diff*index->offsets[dim];
    // TRACE(3,"new offset: %d",index->offset);
}

static inline
int64_t
index_get(index_ctx *index, size_t dim) {
    return index->indices[dim];
}

#endif
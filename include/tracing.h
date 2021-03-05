#ifndef TRACING_H
#define TRACING_H

#ifndef TRACE_SYMBOL_FPRINTF
#include <stdio.h>
#define TRACE_SYMBOL_FPRINTF fprintf
#endif

#ifndef TRACE_SYMBOL_FFLUSH
#include <stdio.h>
#define TRACE_SYMBOL_FFLUSH fflush
#endif

#ifndef TRACE_SYMBOL_ABORT
#include <stdlib.h>
#define TRACE_SYMBOL_ABORT abort
#endif

#ifndef TRACE_SYMBOL_STRTOL
#include <stdlib.h>
#define TRACE_SYMBOL_STRTOL strtol
#endif

#ifndef TRACE_SYMBOL_STDOUT
#define TRACE_SYMBOL_STDOUT stdout
#endif

#ifndef TRACE_SYMBOL_STDERR
#define TRACE_SYMBOL_STDERR stderr
#endif

/** macro TRACE_FLUSH()
 *  flushes TRACE_SYMBOL_STDOUT, TRACE_SYMBOL_STDERR with provided TRACE_SYMBOL_FFLUSH
 *
 *  LEVEL: on which TRACE_LEVEL to start flushing
 **/
#define TRACE_FLUSH(LEVEL) \
_TRACE_FLUSH(LEVEL);

/** macro TRACE_SCOPE(MSG,SAV)
 *  saves old scope to SAV if not NULL and sets MSG as new scope
 *
 *  MSG: string which describes the new scope
 *  SAV: char pointer to save the old scope (NULLable)
 **/
#define TRACE_SCOPE(MSG,SAV) \
_TRACE_SCOPE(MSG,SAV);

/** macro TRACE_ENTRY(LEVEL)
 *  logs entry to current function to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 **/
#define TRACE_ENTRY(LEVEL) \
_TRACE_ENTRY(LEVEL);
/** macro TRACE_EXIT(LEVEL)
 *  logs exit of current function to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 **/
#define TRACE_EXIT(LEVEL) \
_TRACE_EXIT(LEVEL);

/** macro TRACE(LEVEL, COND, FMT, ...)
 *  logs format string FMT with optional variables to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  FMT:   format string
 *  ...:   optional variables
 **/
#define TRACE(LEVEL, COND, FMT, ...) \
_TRACE(LEVEL, COND, FMT, ##__VA_ARGS__);

/** macro TRACE_WARN(LEVEL, COND, FMT, ...)
 *  logs format string FMT with optional variables to TRACE_SYMBOL_STDERR
 *  if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  FMT:   format string
 *  ...:   optional variables
 **/
#define TRACE_WARN(LEVEL, COND, FMT, ...) \
_TRACE_WARN(LEVEL, COND, FMT, ##__VA_ARGS__);

/** macro TRACE_ERROR(LEVEL, COND, FMT, ...)
 *  logs format string FMT with optional variables to TRACE_SYMBOL_STDERR,
 *  flushes afterwards if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  FMT:   format string
 *  ...:   optional variables
 **/
#define TRACE_ERROR(LEVEL, COND, FMT, ...) \
_TRACE_ERROR(LEVEL, COND, FMT, ##__VA_ARGS__);

/** macro TRACE_FATAL(LEVEL, COND, FMT, ...)
 *  logs format string FMT with optional variables to TRACE_SYMBOL_STDERR,
 *  flushes afterwards and aborts execution if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  FMT:   format string
 *  ...:   optional variables
 **/
#define TRACE_FATAL(LEVEL, COND, FMT, ...) \
_TRACE_FATAL(LEVEL, COND, FMT, ##__VA_ARGS__);

/** macro TRACE_BOUND(LEVEL, COND, VAR, MIN, MAX)
 *  logs variable VAR in its boundaries MIN <= VAR < MAX to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  VAR:   variable to check
 *  MIN:   lower boundary of VAR
 *  MAX:   upper boundary of VAR
 **/
#define TRACE_BOUND(LEVEL, COND, VAR, MIN, MAX, FMT) \
_TRACE_BOUND(LEVEL, COND, VAR, MIN, MAX, FMT);

/** macro TRACE_BOUND_WARN(LEVEL, COND, VAR, MIN, MAX)
 *  logs variable VAR in its boundaries MIN <= VAR < MAX to TRACE_SYMBOL_STDOUT,
 *  if TRACE_LEVEL >= LEVEL and COND.
 *  logs boundary exceptions as warning to TRACE_SYMBOL_STDERR and flushes if COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  VAR:   variable to check
 *  MIN:   lower boundary of VAR
 *  MAX:   upper boundary of VAR
 *  FMT:   format string to represent values
 **/
#define TRACE_BOUND_WARN(LEVEL, COND, VAR, MIN, MAX, FMT) \
_TRACE_BOUND_WARN(LEVEL, COND, VAR, MIN, MAX, FMT);

/** macro TRACE_BOUND_ERROR(LEVEL, COND, VAR, MIN, MAX)
 *  logs variable VAR in its boundaries MIN <= VAR < MAX to TRACE_SYMBOL_STDOUT.
 *  if TRACE_LEVEL >= LEVEL and COND.
 *  logs boundary exceptions as error to TRACE_SYMBOL_STDERR and flushes if COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  VAR:   variable to check
 *  MIN:   lower boundary of VAR
 *  MAX:   upper boundary of VAR
 *  FMT:   format string to represent values
 **/
#define TRACE_BOUND_ERROR(LEVEL, COND, VAR, MIN, MAX, FMT) \
_TRACE_ERROR_BOUND(LEVEL, COND, VAR, MIN, MAX, FMT);

/** macro TRACE_BOUND_FATAL(LEVEL, COND, VAR, MIN, MAX)
 *  logs variable VAR in its boundaries MIN <= VAR < MAX to TRACE_SYMBOL_STDOUT.
 *  if TRACE_LEVEL >= LEVEL and COND.
 *  logs boundary exceptions as fatal to TRACE_SYMBOL_STDERR, flushes and aborts
 *  if COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  VAR:   variable to check
 *  MIN:   lower boundary of VAR
 *  MAX:   upper boundary of VAR
 *  FMT:   format string to represent values
 **/
#define TRACE_BOUND_FATAL(LEVEL, COND, VAR, MIN, MAX, FMT) \
_TRACE_BOUND_FATAL(LEVEL, COND, VAR, MIN, MAX, FMT);

/** macro TRACE_VAR(LEVEL, COND, VAR, FMT)
 *  logs variable VAR to TRACE_SYMBOL_STDOUT,
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  VAR:   variable to log
 *  FMT:   format string to represent VAR
 **/
#define TRACE_VAR(LEVEL, COND, VAR, FMT) \
_TRACE_VAR(LEVEL, COND, VAR, FMT);

/** macro TRACE_ARRAY(LEVEL, COND, VAR, NUM, FMT)
 *  logs array VAR to TRACE_SYMBOL_STDOUT,
 *  if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL:   on which TRACE_LEVEL to start logging
 *  COND:    additional condition if logging should happen
 *  VAR:     array to log
 *  ELEMENT: further element specification, can be left empty
 *           i.e. if elements are structs you can specify '.name' to select
 *           the attribute 'name'
 *  NUM:     length of the array
 *  FMT:     format string to represent elements
 **/
#define TRACE_ARRAY(LEVEL, COND, VAR, ELEMENT, NUM, FMT) \
_TRACE_ARRAY(LEVEL, COND, VAR, ELEMENT, NUM, FMT);

/** macro TRACE_ARRAY2D(LEVEL, COND, VAR, NUM_Y, NUM_X, FMT)
 *  logs 2D array VAR to TRACE_SYMBOL_STDOUT,
 *  if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL:   on which TRACE_LEVEL to start logging
 *  COND:    additional condition if logging should happen
 *  VAR:     array to log
 *  ELEMENT: further element specification, can be left empty
 *           i.e. if elements are structs you can specify '.name' to select
 *           the attribute 'name'
 *  NUM_Y:   length of the first dimension
 *  NUM_X:   length of the second dimension
 *  FMT:     format string to represent elements
 **/
#define TRACE_ARRAY2D(LEVEL, COND, VAR, ELEMENT, NUM_Y, NUM_X, FMT) \
_TRACE_ARRAY2D(LEVEL, COND, VAR, ELEMENT, NUM_Y, NUM_X, FMT);

/** macro TRACE_TENSOR(LEVEL, COND, TENSOR)
 *  logs TENSOR to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL:  on which TRACE_LEVEL to start logging
 *  COND:   additional condition if logging should happen
 *  TENSOR: tensor to log
 **/
#define TRACE_TENSOR(LEVEL, COND, TENSOR) \
_TRACE_TENSOR(LEVEL, COND, TENSOR);

/** macro TRACE_ATTRIBUTE(LEVEL, COND, ATTR)
 *  logs ATTR to TRACE_SYMBOL_STDOUT,
 *  if TRACE_LEVEL >= LEVEL and COND
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  ATTR:  attribute to log
 **/
#define TRACE_ATTRIBUTE(LEVEL, COND, ATTR) \
_TRACE_ATTRIBUTE(LEVEL, COND, ATTR);

/** macro TRACE_NODE(LEVEL, COND, NODE)
 *  logs NODE to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  NODE:  node to log
 **/
#define TRACE_NODE(LEVEL, COND, NODE) \
_TRACE_NODE(LEVEL, COND, NODE);

/** macro TRACE_GRAPH(LEVEL, COND, GRAPH)
 *  logs GRAPH to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  GRAPH: node to log
 **/
#define TRACE_GRAPH(LEVEL, COND, GRAPH) \
_TRACE_GRAPH(LEVEL, COND, GRAPH);

/** macro TRACE_MODEL(LEVEL, COND, MODEL)
 *  logs MODEL to TRACE_SYMBOL_STDOUT
 *  if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start logging
 *  COND:  additional condition if logging should happen
 *  MODEL: node to log
 **/
#define TRACE_MODEL(LEVEL, COND, MODEL) \
_TRACE_MODEL(LEVEL, COND, MODEL);

/** macro TRACE_ABORT(LEVEL, COND)
 *  aborts execution if TRACE_LEVEL >= LEVEL and COND.
 *
 *  LEVEL: on which TRACE_LEVEL to start aborting
 *  COND:  additional condition if abort should happen
 **/
#define TRACE_ABORT(LEVEL, COND) \
_TRACE_ABORT(LEVEL, COND);

#ifndef STR
#define STR(X) _STR(X)
#define _STR(X) #X
#endif

#ifndef TRACE_LEVEL

#define __TRACE_COND(LEVEL) ;
#define __PRINT(FD, FMT, ...) ;
#define __FLUSH() ;
#define __PREAMBLE(FD, LEVEL, PREFIX) ;
#define __PREAMBLE_ABORT(FD, LEVEL) ;
#define __ABORT(LEVEL, FMT, ...) ;
#define __PREAMBLE_WARN(FD, LEVEL) ;
#define __WARN(LEVEL, FMT, ...) ;
#define __PREAMBLE_ERROR(FD, LEVEL) ;
#define __ERROR(LEVEL, FMT, ...) ;
#define __PREAMBLE_FATAL(FD, LEVEL) ;
#define __FATAL(LEVEL, FMT, ...) ;
#define __PRINT_VAR(FD, VAR, FMT) ;
#define __VAR(LEVEL, PREFIX, VAR ,FMT) ;
#define __PRINT_ARRAY(FD, VAR, ELEMENT, NUM, FMT) ;
#define __PRINT_ARRAY2D(FD, PREFIX, VAR, ELEMENT, NUM_Y, NUM_X, FMT) ;
#define __PRINT_BOUND(FD, VAR, MIN, MAX, FMT) ;
#define _TRACE(LEVEL, COND, FMT, ...) ;
#define _TRACE_VAR(LEVEL, COND, VAR, FMT) ;
#define _TRACE_ARRAY(LEVEL, COND, VAR, ELEMENT, NUM, FMT) ;
#define _TRACE_ARRAY2D(LEVEL, COND, VAR, ELEMENT, NUM_Y, NUM_X, FMT) ;
#define _TRACE_FLUSH(LEVEL) ;
#define _TRACE_ABORT(LEVEL, COND) ;
#define _TRACE_SCOPE(MSG, SAV) ;
#define _TRACE_ENTRY(LEVEL) ;
#define _TRACE_EXIT(LEVEL)  ;
#define _TRACE_WARN(LEVEL, COND, FMT, ...)  ;
#define _TRACE_ERROR(LEVEL, COND, FMT, ...) ;
#define _TRACE_FATAL(LEVEL, COND, FMT, ...) ;
#define __BOUND(LEVEL, PREFIX, VAR, MIN, MAX, FMT) ;
#define __BOUND_COND(VAR, MIN, MAX) ;
#define __BOUND_WARN(LEVEL, VAR, MIN, MAX, FMT) ;
#define __BOUND_ERROR(LEVEL, VAR, MIN, MAX, FMT) ;
#define __BOUND_FATAL(LEVEL, VAR, MIN, MAX, FMT) ;
#define _TRACE_BOUND(LEVEL, COND, VAR, MIN, MAX, FMT) ;
#define _TRACE_BOUND_WARN(LEVEL, COND, VAR, MIN, MAX, FMT) ;
#define _TRACE_BOUND_ERROR(LEVEL, COND, VAR, MIN, MAX, FMT) ;
#define _TRACE_BOUND_FATAL(LEVEL, COND, VAR, MIN, MAX, FMT) ;
#define _TRACE_TENSOR(LEVEL, COND, TENSOR) ;
#define _TRACE_ATTRIBUTE(LEVEL, COND, ATTR) ;
#define _TRACE_NODE(LEVEL, COND, NODE) ;
#define _TRACE_GRAPH(LEVEL, COND, GRAPH) ;
#define _TRACE_MODEL(LEVEL, COND, MODEL) ;

#else

#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include "onnx.pb-c.h"


__attribute__((unused))
static char *_trace_scope = "";

__attribute__((unused))
static char *_attribute_types[] = {
"UNDEFINED",     // 0
"FLOAT",         // 1
"INT",           // 2
"STRING",        // 3
"TENSOR",        // 4
"GRAPH",         // 5
"FLOATS",        // 6
"INTS",          // 7
"STRINGS",       // 8
"TENSORS",       // 9
"GRAPHS",        // 10
"SPARSE_TENSOR", // 11
"SPARSE_TENSORS" // 12
};

__attribute__((unused))
static int _n_attribute_types = sizeof(_attribute_types)/sizeof(_attribute_types[0]);

__attribute__((unused))
static char *_tensor_types[] = {
"UNDEFINED",  // 0
"FLOAT",      // 1
"UINT8",      // 2
"INT8",       // 3
"UINT16",     // 4
"INT16",      // 5
"INT32",      // 6
"INT64",      // 7
"STRING",     // 8
"BOOL",       // 9
"FLOAT16",    // 10
"DOUBLE",     // 11
"UINT32",     // 12
"UINT64",     // 13
"COMPLEX64",  // 14
"COMPLEX128", // 15
"BFLOAT16"    // 16
};

__attribute__((unused))
static int _n_tensor_types = sizeof(_tensor_types)/sizeof(_tensor_types[0]);

static
int
_trace_severity(const char* overrides, const char *identifier) {
    if ( overrides == NULL || identifier == NULL ) return 0;
    char *start = strstr(overrides,identifier);
    if (!start) return 0;
    char *colon = start + strlen(identifier);
    if (*colon != ':') return 0;
    char *delim = NULL;
    long severity = TRACE_SYMBOL_STRTOL(colon+1,&delim,10);
    if (*delim != ';' && *delim != '\0') return 0;
    return severity;
}

#define __TRACE_COND(LEVEL) \
(((TRACE_LEVEL) >= (LEVEL)) \
|| _trace_severity(getenv("CONNXR_TRACE_FILE"),__FILE__) >= (LEVEL) \
|| _trace_severity(getenv("CONNXR_TRACE_FUNCTION"),__FUNCTION__) >= (LEVEL))


#define __PRINT(FD, FMT, ...) \
TRACE_SYMBOL_FPRINTF(FD, FMT, ##__VA_ARGS__);

#define __FLUSH() \
TRACE_SYMBOL_FFLUSH(TRACE_SYMBOL_STDOUT); \
TRACE_SYMBOL_FFLUSH(TRACE_SYMBOL_STDERR);

#define __PREAMBLE(FD, LEVEL, PREFIX) \
__PRINT(FD, \
        "[%-10.10s%d] " __FILE__ ":%-3d %s ", \
        PREFIX, LEVEL, __LINE__, _trace_scope)

#define __PREAMBLE_ABORT(FD, LEVEL) \
__PREAMBLE(FD, LEVEL, "ABORT")

#define __ABORT(LEVEL, FMT, ...) \
__PREAMBLE_ABORT(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT(TRACE_SYMBOL_STDERR, FMT, ##__VA_ARGS__) \
TRACE_SYMBOL_ABORT();

#define __PREAMBLE_WARN(FD, LEVEL) \
__PREAMBLE(FD, LEVEL, "WARNING")

#define __WARN(LEVEL, FMT, ...) \
__PREAMBLE_WARN(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT(TRACE_SYMBOL_STDERR, FMT, ##__VA_ARGS__) \
__FLUSH()

#define __PREAMBLE_ERROR(FD, LEVEL) \
__PREAMBLE(FD, LEVEL, "ERROR")

#define __ERROR(LEVEL, FMT, ...) \
__PREAMBLE_ERROR(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT(TRACE_SYMBOL_STDERR, FMT, ##__VA_ARGS__) \
__FLUSH()

#define __PREAMBLE_FATAL(FD, LEVEL) \
__PREAMBLE(FD, LEVEL, "FATAL")

#define __FATAL(LEVEL, FMT, ...) \
__PREAMBLE_FATAL(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT(TRACE_SYMBOL_STDERR, FMT, ##__VA_ARGS__) \
__FLUSH() \
__ABORT(LEVEL, "aborting in function %s", __FUNCTION__)

#define __PRINT_VAR(FD, VAR, FMT) \
__PRINT(FD, STR(VAR) ": " FMT, VAR)

#define __VAR(LEVEL, PREFIX, VAR ,FMT) \
__PREAMBLE(TRACE_SYMBOL_STDOUT, LEVEL, PREFIX) \
__PRINT_VAR(TRACE_SYMBOL_STDOUT, VAR, FMT)

#define __PRINT_ARRAY(FD, VAR, ELEMENT, NUM, FMT) \
{ \
    __PRINT(FD, "[") \
    for (int i = 0; i < (NUM); i++) { \
        __PRINT(FD, FMT, VAR[i]ELEMENT) \
        if (i < (NUM)-1) { \
            __PRINT(FD, ",") \
        } \
    } \
    __PRINT(FD, "]") \
}

#define __PRINT_ARRAY2D(FD, LEVEL, PREFIX, VAR, ELEMENT, NUM_Y, NUM_X, FMT) \
{ \
    __PREAMBLE(FD, LEVEL, PREFIX) \
    __PRINT(FD, STR(VAR) ": %p [\n", VAR) \
    for (int y = 0; y < (NUM_Y); y++) { \
        __PREAMBLE(FD, LEVEL, PREFIX) \
        __PRINT(FD, STR(VAR) ": %p  [", &(VAR)[y*(NUM_X)]) \
        for (int x = 0; x < (NUM_X); x++) { \
            __PRINT(FD, FMT, VAR[y*(NUM_X)+x]ELEMENT) \
            if (x < (NUM_X)-1) { \
                __PRINT(FD, ",") \
            } \
        } \
        __PRINT(FD, "]") \
        if (y < (NUM_Y)-1) { \
            __PRINT(FD, ",\n") \
        } \
    } \
    __PRINT(FD, "\n") \
    __PREAMBLE(FD, LEVEL, PREFIX) \
    __PRINT(FD, STR(VAR) ": %p ]\n", &(VAR)[(NUM_Y)*(NUM_X)]) \
}

#define __PRINT_BOUND(FD, VAR, MIN, MAX, FMT) \
__PRINT(FD, \
       FMT " <= " FMT " < " FMT \
       " (" STR(MIN) " <= " STR(VAR) " <= " STR(MAX) ")", \
       MIN, VAR, MAX)

#define _TRACE(LEVEL, COND, FMT, ...) \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __PREAMBLE(TRACE_SYMBOL_STDOUT, LEVEL, "TRACE") \
    __PRINT(TRACE_SYMBOL_STDOUT, FMT "\n", ##__VA_ARGS__) \
}

#define _TRACE_VAR(LEVEL, COND, VAR, FMT) \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __VAR(LEVEL, "VARIABLE", VAR, FMT "\n") \
}

#define _TRACE_ARRAY(LEVEL, COND, VAR, ELEMENT, NUM, FMT) \
if (COND) { \
    if ((VAR) == NULL && (NUM)) { \
        __FATAL(0, "array " STR(VAR) " is %p\n", VAR) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "ARRAY", VAR, "%p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, VAR, ELEMENT, NUM, FMT) \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    } \
}

#define _TRACE_ARRAY2D(LEVEL, COND, VAR, ELEMENT, NUM_Y, NUM_X, FMT) \
if(COND) { \
    if ((VAR) == NULL && (NUM_Y) && (NUM_X)) { \
        __FATAL(0, "array " STR(VAR) " is %p\n", VAR) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __PRINT_ARRAY2D(TRACE_SYMBOL_STDOUT, \
                        LEVEL, \
                        "ARRAY2D", \
                        VAR, \
                        ELEMENT, \
                        NUM_Y, \
                        NUM_X, \
                        FMT) \
    } \
}

#define _TRACE_FLUSH(LEVEL) \
if (__TRACE_COND(LEVEL)) { \
    __FLUSH() \
}

#define _TRACE_ABORT(LEVEL, COND) \
if ((__TRACE_COND(LEVEL)) && (COND)) { \
    __ABORT(LEVEL, "aborting in function %s\n", __FUNCTION__) \
}

#define _TRACE_SCOPE(MSG, SAV) \
{ \
    if (SAV) { (SAV) = _trace_scope; } \
    _trace_scope = (MSG); \
}

#define _TRACE_ENTRY(LEVEL) \
if (__TRACE_COND(LEVEL)) { \
    __PREAMBLE(TRACE_SYMBOL_STDOUT, LEVEL, "ENTRY") \
    __PRINT(TRACE_SYMBOL_STDOUT, "entering function %s\n",__FUNCTION__) \
}

#define _TRACE_EXIT(LEVEL)  \
if (__TRACE_COND(LEVEL)) { \
    __PREAMBLE(TRACE_SYMBOL_STDOUT, LEVEL, "EXIT") \
    __PRINT(TRACE_SYMBOL_STDOUT, "exiting function %s\n",__FUNCTION__) \
}

#define _TRACE_WARN(LEVEL, COND, FMT, ...)  \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __WARN(LEVEL, FMT "\n", ##__VA_ARGS__) \
}

#define _TRACE_ERROR(LEVEL, COND, FMT, ...) \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __ERROR(LEVEL, FMT "\n", ##__VA_ARGS__) \
}

#define _TRACE_FATAL(LEVEL, COND, FMT, ...) \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __FATAL(LEVEL, FMT "\n", ##__VA_ARGS__) \
}

#define __BOUND(LEVEL, PREFIX, VAR, MIN, MAX, FMT) \
__PREAMBLE(TRACE_SYMBOL_STDOUT, LEVEL, PREFIX) \
__PRINT_BOUND(TRACE_SYMBOL_STDOUT, VAR, MIN, MAX, FMT) \
__PRINT(TRACE_SYMBOL_STDOUT, "\n")

#define __BOUND_COND(VAR, MIN, MAX) \
((VAR) >= (MIN) && (VAR) < (MAX))

#define __BOUND_WARN(LEVEL, VAR, MIN, MAX, FMT) \
__PREAMBLE_WARN(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT_BOUND(TRACE_SYMBOL_STDERR, VAR, MIN, MAX, FMT) \
__PRINT(TRACE_SYMBOL_STDERR, "\n") \
__FLUSH()

#define __BOUND_ERROR(LEVEL, VAR, MIN, MAX, FMT) \
__PREAMBLE_ERROR(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT_BOUND(TRACE_SYMBOL_STDERR, VAR, MIN, MAX, FMT) \
__PRINT(TRACE_SYMBOL_STDERR, "\n") \
__FLUSH()

#define __BOUND_FATAL(LEVEL, VAR, MIN, MAX, FMT) \
__PREAMBLE_FATAL(TRACE_SYMBOL_STDERR, LEVEL) \
__PRINT_BOUND(TRACE_SYMBOL_STDERR, VAR, MIN, MAX, FMT) \
__PRINT(TRACE_SYMBOL_STDERR, "\n") \
__FLUSH() \
__ABORT(LEVEL, "aborting in function %s\n", __FUNCTION__)

#define _TRACE_BOUND(LEVEL, COND, VAR, MIN, MAX, FMT) \
if (__TRACE_COND(LEVEL) && (COND)) { \
    __BOUND(LEVEL, "BOUND", VAR, MIN, MAX, FMT) \
}

#define _TRACE_BOUND_WARN(LEVEL, COND, VAR, MIN, MAX, FMT) \
if (COND) { \
    if (__TRACE_COND(LEVEL)) { \
        __BOUND(LEVEL, "BOUND", VAR, MIN, MAX, FMT) \
    } \
    if (!__BOUND_COND(VAR,MIN,MAX)) { \
        __BOUND_WARN(LEVEL, VAR, MIN, MAX, FMT) \
    } \
}

#define _TRACE_BOUND_ERROR(LEVEL, COND, VAR, MIN, MAX, FMT) \
if (COND) { \
    if (__TRACE_COND(LEVEL)) { \
        __BOUND(LEVEL, "BOUND", VAR, MIN, MAX, FMT) \
    } \
    if (!__BOUND_COND(VAR,MIN,MAX)) { \
        __BOUND_ERROR(LEVEL, VAR, MIN, MAX, FMT) \
    } \
}

#define _TRACE_BOUND_FATAL(LEVEL, COND, VAR, MIN, MAX, FMT) \
if (COND) { \
    if (__TRACE_COND(LEVEL)) { \
        __BOUND(LEVEL, "BOUND", VAR, MIN, MAX, FMT) \
    } \
    if (!__BOUND_COND(VAR,MIN,MAX)) { \
        __BOUND_FATAL(LEVEL, VAR, MIN, MAX, FMT) \
    } \
}

#define _TRACE_TENSOR(LEVEL, COND, TENSOR) \
if (COND) { \
    if ((TENSOR) == NULL) { \
        __FATAL(0, "tensor " STR(TENSOR) " is %p\n",TENSOR) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "TENSOR", TENSOR->name, "             \"%s\"\n"); \
        __VAR(LEVEL, "TENSOR", TENSOR->data_type, "        %d" PRId32) \
        if ((TENSOR->data_type) >= _n_tensor_types) { \
            __PRINT(TRACE_SYMBOL_STDOUT, " (%s)\n", _tensor_types[0]) \
            __BOUND_ERROR(0, TENSOR->data_type, 0, _n_tensor_types, "%d" PRId32) \
            __ERROR(0, "unknown data type") \
        } else { \
            __PRINT(TRACE_SYMBOL_STDOUT, " (%s)\n", \
                    _tensor_types[TENSOR->data_type]) \
        } \
        __VAR(LEVEL, "TENSOR", TENSOR->n_dims, "           %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->dims, "             %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      TENSOR->dims, \
                      , \
                      TENSOR->n_dims, \
                      "%" PRId64) \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->has_data_type, "    %d\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->segment, "          %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_float_data, "     %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->float_data, "       %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_int32_data, "     %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->int32_data, "       %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_string_data, "    %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->string_data, "      %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_int64_data, "     %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->int64_data, "       %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->doc_string, "       \"%s\"\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->has_raw_data, "     %d\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->raw_data.len, "     %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->raw_data.data, "    %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_external_data, "  %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->external_data, "    %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->has_data_location, "%d\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->data_location, "    %d\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_double_data, "    %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->double_data, "      %p\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->n_uint64_data, "    %zu\n") \
        __VAR(LEVEL, "TENSOR", TENSOR->uint64_data, "      %p\n") \
    } \
}


#define _TRACE_ATTRIBUTE(LEVEL, COND, ATTR) \
if (COND) { \
    if ((ATTR) == NULL) { \
        __FATAL(0, "attribute " STR(ATTR) " is %p\n",ATTR) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->name, "           \"%s\"\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->ref_attr_name, "  \"%s\"\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->type, "            %d") \
        if ((ATTR->type) >= _n_attribute_types) { \
            __PRINT(TRACE_SYMBOL_STDOUT, " (%s)\n", _attribute_types[0]) \
            __BOUND_ERROR(0, ATTR->type, 0, _n_attribute_types, "%d") \
            __ERROR(0, "unknown data type") \
        } else { \
            __PRINT(TRACE_SYMBOL_STDOUT, " (%s)\n", \
                    _attribute_types[ATTR->type]) \
        } \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->doc_string, "     \"%s\"\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->has_type, "        %d\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->has_f, "           %d\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->f, "               %f\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->has_i, "           %d\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->i, "               % " PRId64 "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->has_s, "           %d\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->s.len, "           %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->s.data, "          %p ") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\"%.*s\"\n", (int)ATTR->s.len, ATTR->s.data) \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->t, "               %p\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->g, "               %p\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->sparse_tensor, "   %p\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_floats, "        %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->floats, "          %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      ATTR->floats, \
                      , \
                      ATTR->n_floats, \
                      "%f") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_ints, "          %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->ints, "            %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      ATTR->ints, \
                      , \
                      ATTR->n_ints, \
                      "%" PRId64) \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_strings, "       %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->strings, "         %p ") \
        { \
            __PRINT(TRACE_SYMBOL_STDOUT, "[") \
            for (int i = 0; i < (ATTR)->n_strings; i++) { \
                __PRINT(TRACE_SYMBOL_STDOUT, "\"%.*s\"", \
                        (int)ATTR->strings[i].len, ATTR->strings[i].data) \
                if (i < (ATTR)->n_strings-1) { \
                    __PRINT(TRACE_SYMBOL_STDOUT, ",") \
                } \
            } \
            __PRINT(TRACE_SYMBOL_STDOUT, "]") \
        } \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_tensors, "       %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->tensors, "         %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      ATTR->tensors, \
                      , \
                      ATTR->n_tensors, \
                      "%p") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_graphs, "        %zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->graphs, "          %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      ATTR->graphs, \
                      , \
                      ATTR->n_graphs, \
                      "%p") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->n_sparse_tensors, "%zu\n") \
        __VAR(LEVEL, "ATTRIBUTE", ATTR->sparse_tensors, "  %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      ATTR->sparse_tensors, \
                      , \
                      ATTR->n_sparse_tensors, \
                      "%p") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    } \
}

#define _TRACE_NODE(LEVEL, COND, NODE) \
if (COND) {\
    if ((NODE) == NULL) { \
        __FATAL(0, "node " STR(NODE) " is %p\n",NODE) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "NODE", NODE->domain, "     \"%s\"\n") \
        __VAR(LEVEL, "NODE", NODE->name, "       \"%s\"\n") \
        __VAR(LEVEL, "NODE", NODE->op_type, "    \"%s\"\n") \
        __VAR(LEVEL, "NODE", NODE->n_input, "    %zu\n") \
        __VAR(LEVEL, "NODE", NODE->input, "      %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      NODE->input, \
                      , \
                      NODE->n_input, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "NODE", NODE->n_output, "   %zu\n") \
        __VAR(LEVEL, "NODE", NODE->output, "     %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      NODE->output, \
                      , \
                      NODE->n_output, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "NODE", NODE->n_attribute, "%zu\n") \
        __VAR(LEVEL, "NODE", NODE->attribute, "  %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      NODE->attribute, \
                      ->name, \
                      NODE->n_attribute, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "NODE", NODE->doc_string, " \"%s\"\n") \
    } \
}

#define _TRACE_GRAPH(LEVEL, COND, GRAPH) \
if (COND) { \
    if ((GRAPH) == NULL) { \
        __FATAL(0, "graph " STR(GRAPH) " is %p\n",GRAPH) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "GRAPH", GRAPH->name, "                     \"%s\"\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_node, "                   %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->node, "                     ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->node, \
                      ->name, \
                      GRAPH->n_node, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_initializer, "            %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->initializer, "              ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->initializer, \
                      ->name, \
                      GRAPH->n_initializer, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_sparse_initializer, "     %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->sparse_initializer, "       ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->sparse_initializer, \
                      , \
                      GRAPH->n_sparse_initializer, \
                      "%p") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->doc_string, "               \"%s\"") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_input, "                  %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->input, "                    ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->input, \
                      ->name, \
                      GRAPH->n_input, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_output, "                 %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->output, "                   ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->output, \
                      ->name, \
                      GRAPH->n_output, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_value_info, "             %zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->value_info, "               ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->value_info, \
                      ->name, \
                      GRAPH->n_value_info, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->n_quantization_annotation, "%zu\n") \
        __VAR(LEVEL, "GRAPH", GRAPH->quantization_annotation, "  ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      GRAPH->quantization_annotation, \
                      ->tensor_name, \
                      GRAPH->n_quantization_annotation, "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    } \
}

#define _TRACE_MODEL(LEVEL, COND, MODEL) \
if (COND) { \
    if ((MODEL) == NULL) { \
        __FATAL(0, "model " STR(MODEL) " is %p\n",MODEL) \
    } \
    if (__TRACE_COND(LEVEL)) { \
        __VAR(LEVEL, "MODEL", MODEL->has_ir_version, "   %d\n") \
        __VAR(LEVEL, "MODEL", MODEL->ir_version, "       %" PRId64 "\n") \
        __VAR(LEVEL, "MODEL", MODEL->n_opset_import, "   %zu\n") \
        __VAR(LEVEL, "MODEL", MODEL->opset_import, "     %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      MODEL->opset_import, \
                      ->domain, \
                      MODEL->n_opset_import, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
        __VAR(LEVEL, "MODEL", MODEL->producer_name, "    \"%s\"\n") \
        __VAR(LEVEL, "MODEL", MODEL->producer_version, " \"%s\"\n") \
        __VAR(LEVEL, "MODEL", MODEL->domain, "           \"%s\"\n") \
        __VAR(LEVEL, "MODEL", MODEL->has_model_version, "%d\n") \
        __VAR(LEVEL, "MODEL", MODEL->model_version, "    %" PRId64 "\n") \
        __VAR(LEVEL, "MODEL", MODEL->doc_string, "       \"%s\"\n") \
        __VAR(LEVEL, "MODEL", MODEL->graph, "            %p\n") \
        __VAR(LEVEL, "MODEL", MODEL->n_metadata_props, " %zu\n") \
        __VAR(LEVEL, "MODEL", MODEL->metadata_props, "   %p ") \
        __PRINT_ARRAY(TRACE_SYMBOL_STDOUT, \
                      MODEL->metadata_props, \
                      ->key, \
                      MODEL->n_metadata_props, \
                      "\"%s\"") \
        __PRINT(TRACE_SYMBOL_STDOUT, "\n") \
    } \
}

#endif

#endif
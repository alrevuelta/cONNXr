#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "conv.h"

// TODO

// TODO Use one file per operator

/* Important Notes
 *  If you are contributing by implementing an operator, make sure that you follow
 *  onnx specifications in https://github.com/onnx/onnx/blob/master/docs/Operators.md
 *  Remember to copy the documentation, like inputs, outputs and type constraints.
 *  See previosly implemented operators as example.
 */

// Template example
/*! \fn COPY_PASTE_FUNCTION_DECLARATION
 *  \brief COPY_PASTE_AND_FORMAT_ONNX_DOCUMENTATION. INPUTS/OUTPUTS/CONSTRAINTS
 *
 *         Limitations: There might be some limitations with respect to the onnx
 *           official operator. Write here possible limitations, i.e. if the
 *           function doesnt work with all types, or if it works with a specific
 *           number of dimensions only
 *  \param[in]  xx xx
 *  \param[in]  xx xx
 *  \param[out] xx xx
 *  \return     xx
 */


 // Template supported Types
 /*
 switch(type)
 {
   case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
     break;
   case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
     break;
   default:
     break;
 }*/

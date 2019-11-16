#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "maxpool.h"

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
 void operator_maxpool(Onnx__TensorProto *X,
                       Onnx__TensorProto *Y,
                       Onnx__TensorProto *Indices,
                       size_t n_attribute,
                       Onnx__AttributeProto **attribute)
{
  debug_print_attributes(n_attribute, attribute);
  // TODO check this? no mem is allocated?
  Y->name         = "name_is_set_afterwards\0";
  // Allocte memory

  /*Y->dims = malloc(xxx * sizeof(int64_t));

  // Populate some parameters
  Y->name         = "name_is_set_afterwards\0";
  Y->n_dims       = xxx;
  Y->dims[xx]      = xxx
  Y->has_raw_data = 0;*/

  switch(X->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    default:
      break;
  }

}

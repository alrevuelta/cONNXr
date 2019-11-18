#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../pb/onnx.pb-c.h"
#include "../embeddedml_debug.h"
#include "conv.h"

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
 void operator_conv(Onnx__TensorProto *X,
                    Onnx__TensorProto *W,
                    Onnx__TensorProto *B,
                    Onnx__TensorProto *Y,
                    size_t n_attribute,
                    Onnx__AttributeProto **attribute)
{
  DEBUG_PRINT("Calling operator_conv");
  // Borrowed form https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/convolutional_layer.c#L445
  // TODO dilations is harcoded [1 1]
  // TODO strides is hardcoded [1 1]
  // TODO group is hardcoded 1

  printf("X->name = %s", X->name);
  printf("W->name = %s", W->name);

  switch(X->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
    {

    }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      break;
    default:
      break;
  }
}

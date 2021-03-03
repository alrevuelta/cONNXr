#include "operators/operator_info.h"

#include <string.h>

char*
operator_info_attributeType2str(uint32_t type)
{
  switch (type) {
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED:
      return "UNDEFINED";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
      return "FLOAT";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
      return "INT";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
      return "STRING";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
      return "TENSOR";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH:
      return "GRAPH";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
      return "SPARSE_TENSOR";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
      return "FLOATS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
      return "INTS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
      return "STRINGS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS:
      return "TENSORS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS:
      return "GRAPHS";
    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS:
      return "SPARSE_TENSORS";
    default:
      return NULL;
  }
}

char*
operator_info_tensorType2str(uint32_t type)
{
  switch (type) {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      return "UNDEFINED";
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      return "FLOAT";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      return "UINT8";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      return "INT8";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      return "UINT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      return "INT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      return "INT32";
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      return "INT64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      return "STRING";
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      return "BOOL";
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      return "FLOAT16";
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      return "DOUBLE";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      return "UINT32";
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      return "UINT64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      return "COMPLEX64";
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      return "COMPLEX128";
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      return "BFLOAT16";
    default:
      return NULL;
  }
}
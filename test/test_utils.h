#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>

#define FLOAT_TOLERANCE 0.00001f

// Compare if equal with some tolarenace
void compareAlmostEqualTensorProto(Onnx__TensorProto *a, Onnx__TensorProto *b)
{

  CU_ASSERT_EQUAL(a->data_type, b->data_type);
  DEBUG_PRINT("a->ndims %zu, b->n_dims %zu", a->n_dims, b->n_dims);
  CU_ASSERT_EQUAL(a->n_dims, b->n_dims);
  for (int d = 0; d < a->n_dims; d++)
  {
    CU_ASSERT_EQUAL(a->dims[d], b->dims[d]);
  }

  // TODO Not all types are implemented
  switch(a->data_type)
  {
    case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
      CU_FAIL("Data types is undefined");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
      CU_ASSERT_EQUAL(a->n_float_data, b->n_float_data);
      for(int i = 0; i < a->n_float_data; i++)
      {
        //CU_ASSERT_EQUAL(a->float_data[i], b->float_data[i]);
        DEBUG_PRINT("%f, %f", a->float_data[i], b->float_data[i]);
        CU_ASSERT(fabs(a->float_data[i] - b->float_data[i]) < FLOAT_TOLERANCE);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
      CU_FAIL("uint8 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
      CU_FAIL("int8 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
      CU_FAIL("uint16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
      CU_FAIL("int16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
      CU_FAIL("int32 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
      CU_ASSERT_EQUAL(a->n_int64_data, b->n_int64_data);
      for(int i = 0; i < a->n_int64_data; i++)
      {
        DEBUG_PRINT("ASSERTING EQUAL: %lld, %lld", a->int64_data[i], b->int64_data[i]);
        CU_ASSERT_EQUAL(a->int64_data[i], b->int64_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
      CU_FAIL("string data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
      CU_FAIL("bool data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
      CU_FAIL("float16 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
      CU_ASSERT_EQUAL(a->n_double_data, b->n_double_data);
      for(int i = 0; i < a->n_double_data; i++)
      {
        DEBUG_PRINT("ASSERTING EQUAL: %lf, %lf", a->double_data[i], b->double_data[i]);
        CU_ASSERT_EQUAL(a->double_data[i], b->double_data[i]);
      }
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
      CU_FAIL("uint32 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
      CU_FAIL("uint64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
      CU_FAIL("complex64 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
      CU_FAIL("complex128 data_type is not implemented");
      break;
    case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
      CU_FAIL("bfloat16 data_type is not implemented");
      break;
    default:
      CU_FAIL("Unknown data_type");
      break;
  }
}

#endif

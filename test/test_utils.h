#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>

int compareEqualTensorProto(Onnx__TensorProto *a, Onnx__TensorProto *b)
{
  int areEq = 0;

  CU_ASSERT(a->n_dims == b->n_dims);

  // TODO Assert dims[n]

  CU_ASSERT(a->data_type == b->data_type);

  CU_ASSERT(a->n_float_data == b->n_float_data);

  for(int i = 0; i < a->n_float_data; i++)
  {
    CU_ASSERT(a->float_data[i] = b->float_data[i]);
  }

  // TODO Assert variable type depending on data_type

  return 0;
}

#endif

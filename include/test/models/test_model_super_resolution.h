#ifndef TEST_MODEL_SUPER_RESOLUTION_H
#define TEST_MODEL_SUPER_RESOLUTION_H

#include "common_models.h"

void test_model_super_resolution(void)
{
  test_model(
    "super_resolution",
    "test/super_resolution/super_resolution.onnx",
    "test/super_resolution/test_data_set_0",
    1,
    1);
}

#endif

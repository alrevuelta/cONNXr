#ifndef TEST_MODEL_MOBILENETV2_H
#define TEST_MODEL_MOBILENETV2_H

#include "common_models.h"

void test_model_mobilenetv2(void)
{
  /* This model comes with three testing vectors (0,1,2) but only
  0 is used, since using all of them will take quite long to run*/
  test_model(
    "mobilenetv2",
    "test/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
    "test/mobilenetv2-1.0/test_data_set_0",
    1,
    1);
}

#endif

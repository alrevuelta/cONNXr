#ifndef TEST_MODEL_TINYYOLOV2_H
#define TEST_MODEL_TINYYOLOV2_H

#include "common_models.h"

void test_model_tinyyolov2(void)
{
  /* This model comes with three testing vectors (0,1,2) but only
  0 is used, since using all of them will take quite long to run*/
  test_model(
    "tinyyolov2",
    "test/tiny_yolov2/Model.onnx",
    "test/tiny_yolov2/test_data_set_0",
    1,
    1);
}

#endif

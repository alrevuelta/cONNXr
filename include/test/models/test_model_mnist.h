#ifndef TEST_MODEL_MNIST_H
#define TEST_MODEL_MNIST_H

#include "common_models.h"

void test_model_mnist(void)
{
  test_model("mnist", "test/mnist/model.onnx",
             "test/mnist/test_data_set_0", 1, 1);
  test_model("mnist", "test/mnist/model.onnx",
             "test/mnist/test_data_set_1", 1, 1);
  test_model("mnist", "test/mnist/model.onnx",
             "test/mnist/test_data_set_2", 1, 1);
}

#endif

# Tests

* Use cunit to write unit tests
* Explore if Python can be used to test the operators. Onnx defines most of the operators using a numpy function as reference.
* `node` folder is directly downloaded from onnx repo, "backend/test/data". It provides testcases for a onnx runtime at two different levels. Node level (invidual operations such as sigmoid, matmul) and full graph, which is a real ML model.

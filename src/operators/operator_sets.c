
//this file was generated by ../../scripts/onnx_generator/OperatorSets.py
#include "operators/operator_sets.h"

#include "operators/onnx/operator__onnx__softmax__11.h"
#include "operators/onnx/operator__onnx__transpose__1.h"
#include "operators/onnx/operator__onnx__relu__6.h"
#include "operators/onnx/operator__onnx__mul__7.h"
#include "operators/onnx/operator__onnx__sigmoid__6.h"
#include "operators/onnx/operator__onnx__reshape__5.h"
#include "operators/onnx/operator__onnx__conv__11.h"
#include "operators/onnx/operator__onnx__leakyrelu__6.h"
#include "operators/onnx/operator__onnx__batchnormalization__9.h"
#include "operators/onnx/operator__onnx__argmax__12.h"
#include "operators/onnx/operator__onnx__matmul__9.h"
#include "operators/onnx/operator__onnx__add__7.h"
#include "operators/onnx/operator__onnx__constant__12.h"
#include "operators/onnx/operator__onnx__maxpool__12.h"

operator_set operator_set__onnx__1 = {
  .version = 1,
  .domain  = "onnx",
  .length  = 1,
  .entries = {
    {
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
}
  }
};

operator_set operator_set__onnx__5 = {
  .version = 5,
  .domain  = "onnx",
  .length  = 2,
  .entries = {
    {
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
}
  }
};

operator_set operator_set__onnx__6 = {
  .version = 6,
  .domain  = "onnx",
  .length  = 5,
  .entries = {
    {
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Relu",
  .resolver = (operator_resolver) &resolve_operator__onnx__relu__6
},{
  .name = "Sigmoid",
  .resolver = (operator_resolver) &resolve_operator__onnx__sigmoid__6
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
},{
  .name = "LeakyRelu",
  .resolver = (operator_resolver) &resolve_operator__onnx__leakyrelu__6
}
  }
};

operator_set operator_set__onnx__7 = {
  .version = 7,
  .domain  = "onnx",
  .length  = 7,
  .entries = {
    {
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Relu",
  .resolver = (operator_resolver) &resolve_operator__onnx__relu__6
},{
  .name = "Mul",
  .resolver = (operator_resolver) &resolve_operator__onnx__mul__7
},{
  .name = "Sigmoid",
  .resolver = (operator_resolver) &resolve_operator__onnx__sigmoid__6
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
},{
  .name = "LeakyRelu",
  .resolver = (operator_resolver) &resolve_operator__onnx__leakyrelu__6
},{
  .name = "Add",
  .resolver = (operator_resolver) &resolve_operator__onnx__add__7
}
  }
};

operator_set operator_set__onnx__9 = {
  .version = 9,
  .domain  = "onnx",
  .length  = 9,
  .entries = {
    {
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Relu",
  .resolver = (operator_resolver) &resolve_operator__onnx__relu__6
},{
  .name = "Mul",
  .resolver = (operator_resolver) &resolve_operator__onnx__mul__7
},{
  .name = "Sigmoid",
  .resolver = (operator_resolver) &resolve_operator__onnx__sigmoid__6
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
},{
  .name = "LeakyRelu",
  .resolver = (operator_resolver) &resolve_operator__onnx__leakyrelu__6
},{
  .name = "BatchNormalization",
  .resolver = (operator_resolver) &resolve_operator__onnx__batchnormalization__9
},{
  .name = "MatMul",
  .resolver = (operator_resolver) &resolve_operator__onnx__matmul__9
},{
  .name = "Add",
  .resolver = (operator_resolver) &resolve_operator__onnx__add__7
}
  }
};

operator_set operator_set__onnx__11 = {
  .version = 11,
  .domain  = "onnx",
  .length  = 11,
  .entries = {
    {
  .name = "Softmax",
  .resolver = (operator_resolver) &resolve_operator__onnx__softmax__11
},{
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Relu",
  .resolver = (operator_resolver) &resolve_operator__onnx__relu__6
},{
  .name = "Mul",
  .resolver = (operator_resolver) &resolve_operator__onnx__mul__7
},{
  .name = "Sigmoid",
  .resolver = (operator_resolver) &resolve_operator__onnx__sigmoid__6
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
},{
  .name = "Conv",
  .resolver = (operator_resolver) &resolve_operator__onnx__conv__11
},{
  .name = "LeakyRelu",
  .resolver = (operator_resolver) &resolve_operator__onnx__leakyrelu__6
},{
  .name = "BatchNormalization",
  .resolver = (operator_resolver) &resolve_operator__onnx__batchnormalization__9
},{
  .name = "MatMul",
  .resolver = (operator_resolver) &resolve_operator__onnx__matmul__9
},{
  .name = "Add",
  .resolver = (operator_resolver) &resolve_operator__onnx__add__7
}
  }
};

operator_set operator_set__onnx__12 = {
  .version = 12,
  .domain  = "onnx",
  .length  = 14,
  .entries = {
    {
  .name = "Softmax",
  .resolver = (operator_resolver) &resolve_operator__onnx__softmax__11
},{
  .name = "Transpose",
  .resolver = (operator_resolver) &resolve_operator__onnx__transpose__1
},{
  .name = "Relu",
  .resolver = (operator_resolver) &resolve_operator__onnx__relu__6
},{
  .name = "Mul",
  .resolver = (operator_resolver) &resolve_operator__onnx__mul__7
},{
  .name = "Sigmoid",
  .resolver = (operator_resolver) &resolve_operator__onnx__sigmoid__6
},{
  .name = "Reshape",
  .resolver = (operator_resolver) &resolve_operator__onnx__reshape__5
},{
  .name = "Conv",
  .resolver = (operator_resolver) &resolve_operator__onnx__conv__11
},{
  .name = "LeakyRelu",
  .resolver = (operator_resolver) &resolve_operator__onnx__leakyrelu__6
},{
  .name = "BatchNormalization",
  .resolver = (operator_resolver) &resolve_operator__onnx__batchnormalization__9
},{
  .name = "ArgMax",
  .resolver = (operator_resolver) &resolve_operator__onnx__argmax__12
},{
  .name = "MatMul",
  .resolver = (operator_resolver) &resolve_operator__onnx__matmul__9
},{
  .name = "Add",
  .resolver = (operator_resolver) &resolve_operator__onnx__add__7
},{
  .name = "Constant",
  .resolver = (operator_resolver) &resolve_operator__onnx__constant__12
},{
  .name = "MaxPool",
  .resolver = (operator_resolver) &resolve_operator__onnx__maxpool__12
}
  }
};

operator_sets all_operator_sets = {
  .length = 7,
  .sets   = {
    &operator_set__onnx__1,
&operator_set__onnx__5,
&operator_set__onnx__6,
&operator_set__onnx__7,
&operator_set__onnx__9,
&operator_set__onnx__11,
&operator_set__onnx__12
  }
};

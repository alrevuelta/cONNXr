# 02 Code Overview

## File structure
You will find all the source code inside `src`. There is also a folder with all the operators `operators`. The idea is to have one file per operator, and every operator should match a common defined interface (inputs/outputs)

TODO:
```
|__examples
|  |__example1
|__scripts
|__src
|  |__operators
|  |__xxx
|__test
```

## Operators interface
In order to standardise the operators implementation, an interface has to be satisfied by each operator. Its just a set of inputs and outputs that all operators have. The operator has 6 inputs, `n_input` represents the size of the array `input`, `n_attribute` indicates the size of `attribute` and `n_output` the size of `output`. The name is pretty much self explanatory, but `input` is the inputs to the node, which are the inputs that will be given to the operator. The `attribute` field indicates some attributes that an operator might have and is operator specific. All these values are inputs but `output` that will be used as a output by reference. Last but not least and `int` is returned indicating if there was any error.

```c
int operator_xxx(const size_t n_input,
                 const Onnx__TensorProto **input,
                 const size_t n_attribute,
                 const Onnx__AttributeProto **attribute,
                 const size_t n_output,
                 Onnx__TensorProto **output);
```

## Protocol Buffers
`onnx` uses protocol buffers to serialize the models data. Note that `protobuf-c` is used to generate the `pb/onnx.pb-c.c` and `pb/onnx.pb-c.h`. Files are already provided, but you can generate it like this:

```
protoc --c_out=. onnx.proto
```

In the future `nanopb` might be used, since it can generate smaller files. Investigate also how to use `.option` file. You can find some initial tests in `pb/nanopb` but is not yet being used. You can regenerate it using the following command, but note that you need to have a `protoc` binary.

```
generator-bin/protoc --nanopb_out=. onnx.proto
```

Note that nanopb achieves a signifiant reduction of the '.c' and `.h` files. 69K/45K for non nanopb and 14/20KB for nanopb. So (69+45)/(14+20) thats 3 times less!

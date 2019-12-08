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

## Types and Structures
If you go through the code, you will see that some custom defined types are used such as `Onnx__TensorProto`. Well, actually they are not custom, but defined by `onnx` in the `.proto` file. You can see them all in [src/pb](src/pb) folder. For example, `Onnx__ModelProto` struct stores the whole model, that contains some information plus a `Onnx__GraphProto`. And inside the graph, there are many `Onnx__NodeProto` connected to each other that take some inputs and apply a specific operator to calculate an output.

One of the most important types that you will see, is the `Onnx__TensorProto`. It just defines a vector, an array, a matrix, or whatever you want to call it. It is quite convinient to use, because it is quite generic. You can store different types of values, with different sizes. As an example, lets say that we want to store a 3 dimension vector. In that case `n_dims=3` and `dims[0]`, `dims[1]`, `dims[2]` will store some values. Lets store some `float` values, so `data_type=ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT `. In this case `n_float_data=dims[0]*dims[1]*dims[2]` and `float_data` will contain all the values in a single dimension array. The tensor has also a `name`.

```c
struct  _Onnx__TensorProto
{
  ProtobufCMessage base;
  size_t n_dims;
  int64_t *dims;

  protobuf_c_boolean has_data_type;
  int32_t data_type;
  Onnx__TensorProto__Segment *segment;

  size_t n_float_data;
  float *float_data;

  size_t n_int32_data;
  int32_t *int32_data;

  size_t n_string_data;
  ProtobufCBinaryData *string_data;

  size_t n_int64_data;
  int64_t *int64_data;

  char *name;
  char *doc_string;

  protobuf_c_boolean has_raw_data;
  ProtobufCBinaryData raw_data;

  size_t n_external_data;
  Onnx__StringStringEntryProto **external_data;

  protobuf_c_boolean has_data_location;
  Onnx__TensorProto__DataLocation data_location;

  size_t n_double_data;
  double *double_data;

  size_t n_uint64_data;
  uint64_t *uint64_data;
};
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

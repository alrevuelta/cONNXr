# 03 Contributing
This project is in very early state so we are looking for contributors. Feel free to open a PR or an issue with questions or suggestions. We would be also happy to guide you if you have an idea. You can contribute in many ways, but the most common is to implement a new operator. The `onnx` specification defines more than 150 operators, that are supported for different types (float, int, double,...) and for some of them there are different versions. Some common ones are implemented with limited functionality like convolutions, matrix multiplications and so on, but there are lots of them remaining.

You can also contribute by improving an existing operator or fixing some bugs. Feel free to have a look to the opened issues, where we also have some simple issues for newcomers.

## Add new operator

If you want to add a new operator, we provide a simple and generic interface that you can use, with all the information in place ready to be used. The interface is the following. You can check [here](https://github.com/alrevuelta/cONNXr/blob/master/doc/01_Documentation.md#types-and-structures) more information about each struct. In `onnx_node`, `inputs` and `outputs` you will have all the information that you need such as the inputs of the operators with its names, the attributes and where to write the output. Refer to [ONNX Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md) to check the inputs/outputs/attributes of the operator you are implementing. Note that some might be optional.
```c
struct node_context{
  Onnx__NodeProto     *onnx_node;
  Onnx__TensorProto  **inputs;
  Onnx__TensorProto  **outputs;
  operator_executer resolved_op;
};
```

First of all decide the operator that you want to implement. You can see the [official onnx operators list](https://github.com/onnx/onnx/blob/master/docs/Operators.md). Lets say that you want to implement the `Abs` operator, fair enough.

Since each operator has different versions, you would need to choose the version that you are implementing. The differences can be rather tiny, but its not the same [Abs-13](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Abs-13) operator than [Abs-6](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Abs-6). Note that the number is the onnx version that introduced that operator. We recommend implementing the latest one available at the moment.

Once you know the operator and version, there is one extra "dimension" to take into account. Most of the operators work with different data types, so you must choose which one to implement. Going back to the `Conv` example, we would suggest to start implementing the `float` version. Note that `Conv` is defined for double, float and float16. The code would be of course quite similar, but since we are in C we can't just use the same functions for all types.

Lets see how you can implement a new operator in few simple steps.

### 1. Generate the files

You don't have to create any files, just do a small modification in the `Makefile` and run a script. The files that you need will be generated automatically:

* If you want to implement a new operator, go to the makefile and add a new line with the operator that you want to generate, i.e. ONNX_INCLUDE+="^Add$$".
* Once you have that, and assuming that you have the latest onnx Python version installed, run `make onnx_generator`.
* The previous step will generate all files that are needed. It will also update the operator_sets.c file and create a resolver for that operator. You don't really need to care about this.
* Now you can populate the .c files with your implementation. Note that there is one implementation per data type (i.e. float, double,...)

Lets see some of the files that `Add` operator has:

* `operator__onnx__add__7__T_tensor_double.c`: The `7` indicates the operator version and `double` indicates the type. Add operator will have other files such as `_int32` or `_int64`. The implementations would be different, but they can share most of the code. You have of course to write in these file the actual implementation of the operator.
* `resolve_operator__onnx__add__7.c`: You don't need to touch this file. It justs maps (resolves) the function that an operator needs based on the input time.


### 2. Implement the operator

Now that you have generated all the files you are ready to start implementing the operator. Lets say that you want to implement `operator__onnx__add__7__T_tensor_float.c`, which is the `Add` operator for opset version `11` and type `float`.

This operator is adding two values or tensors so first of all you need these values. According to [the specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add) this operator takes two inputs `A` and `B` with no attributes. There are two different ways that you can access them: by index or by name.


#### Inputs
You can **access by index** as follows. Note that `ctx->onnx_node->input[n]` doesn't contain the tensor, but just a name of the tensor.
```c
// Access by index
Onnx__TensorProto *A = ctx->inputs[0];
Onnx__TensorProto *B = ctx->inputs[1];
```

You can also **access by name** with `searchInputByName()`. This function takes an index with the name of the tensor that wants to be accessed.

```c
// Access by name
Onnx__TensorProto *A = searchInputByName(ctx, 0);
Onnx__TensorProto *B = searchInputByName(ctx, 1);
```

Both ways are perfectly valid, but the **access by index** is preferred since it doesn't need to search. But note that in some cases it can't be used. Imagine an operator with two inputs, but the second one is optional. Using `ctx->inputs[1]` will only work if the second input is provided, and will fail when it doesn't. For this cases, use `searchInputByName()` that will return NULL.

**Note**: There is some ongoing work in here, so might change the way it works.


At this point you have `A` and `B` ready to be added. Both variables belong to `Onnx__TensorProto` so feel free to have a look [here](https://github.com/alrevuelta/cONNXr/blob/master/doc/01_Documentation.md#types-and-structures). You can access the elements of the tensor like this, assuming that the type stored in it is `float`.

```c
for (int i = 0; i < A->n_float_data){
	//A->float_data[i]
}
```

#### Attributes
If the operator you are implementing has some attributes, you can also easily get them with. Just replace `auto_pad` by your attribute name.

```c
Onnx__AttributeProto *auto_pad = searchAttributeNyName(
  ctx->onnx_node->n_attribute,
  ctx->onnx_node->attribute,
  "auto_pad");
```

Same than before applies here. You can access the attributes **by name** or **by index**

Access by index:
* If there is only 1 attribute, and that attribute is mandatory, this way can be used. Not the case of `LeakyRelu` because that attribute could be empty (and the default value will be taken).
* If there are more than 1 attribute but all of them are mandatory, we can also use this way. If some attributes are mandatory this can't be done.
```c
// Access by index
Onnx__AttributeProto *a_alpha = ctx->onnx_node->attribute[0]->f;
```

And this other way can be used in the rest of the cases.
```c
// Access by name
Onnx__AttributeProto *a_alpha = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute, "alpha");
if (a_alpha) {
     alpha = a_alpha->f;
}
``` 

#### Outputs
Last but not least, you will need to store the result in a variable, so that other nodes can reuse that output. Just use the following function and populate the content.

You can do it also **by index**. This is the way to go if there if only one output that is mandatory.
```c
Onnx__TensorProto *C = ctx->outputs[0];
```

And in more complex cases where you can optional outputs or more than one, you can use the following.
```c
Onnx__TensorProto *C = searchOutputByName(ctx, 0);
```

### 3. Test the operator

Once the operator is implemented, you are ready to test it. Luckily, onnx provides a set of [test vectors](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node) for all operators, and we have taken care of integrating them, so you just need to go to `tests.c` file and uncomment the test case of your operator. For example, if you have implemented the `Shrink` operator, you should uncomment `test_shrink_hard` and `test_shrink_hard`. Its important to note that this test cases are not testing all data types (float, double,...). Most of these test cases run on float, but not always.

Currently `onnx` only provides test cases for the latest operator, so if you want to test and old version of an operator, you will have to do some manual work. However, there is some ongoing work in [here](https://github.com/onnx/onnx/issues/2912) to address this issue.

You can have a look to [this PR](https://github.com/alrevuelta/cONNXr/pull/43) that contains an example of how an operator can be implemented.

## Other contributions
You can also contribute in other ways, like improving an existing operator, fixing bugs or writing documentation.

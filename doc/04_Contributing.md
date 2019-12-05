# 04 Contributing
This repository is in a very early stage, so you can contribute in many ways. Feel free to have a look to [README](README.md) file, where you will find a list with some tasks that you are free to pull. Feel free to go through the documentation first, specifically [02_CodeOverview](doc/02_CodeOverview)

## Add new operator
One of the easiest ways to contribute, is by adding a new operator. Currently, the official `onnx` repository provides more than 150 operators, but not even 10 are implemented. We would love to see a pull request with a new operator :smiley:. It would be great if you can also follow some guidelines:
* Follow the same naming convenction `operator_xxx` and the function declaration. You can see other operators for reference, just declare your function with the same number of inputs and use them.
* Custom operators are allowed, but you might to start first with the ones supported by the official `onnx` repository. You can find them all [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
* We would also love to see some testcases for the new operator. In the `test` folder you have all you need. Feel free to read also [03_Testing](doc/03_Testing). You should cover at least the official tests for a [onnx backed](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node). For example, if you implement `Shrink` operator, you should inculde tests for `test_shrink_hard` and `test_shrink_hard`. Don't worry, we provide some functions that can open both the `.pb` and `.onnx` files easily.
* In spite of that `onnx` provides some backend test cases, we think is not enough, so extra tests are welcome. For example, with `onnx` tests, not all types of inputs are tests (i.e. `float`, `double`,...)

## Modify code
You can also contribute in many ways that are not related to the operators. Improve existing code, new ideas, fix bugs, add tests, whatever you want, we would be please to see your PR.

## Other
There are many other ways you can contribute that might be not code related. Maybe some diagrams, a logo for the project, more documentation, CI loops, ideas, or even feature requests that other people can implement.

# 02 Code Overview

## xx
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

## Protocol Buffers
`onnx` uses protocol buffers to serialize the models data. Note that `protobuf-c` is used to generate the `pb/onnx.pb-c.c` and `pb/onnx.pb-c.h`. Files are already provided, but you can generate it like this:

```
protoc --c_out=. onnx.proto
```

In the future `nanopb` might be used, since it can generate smaller files. Investigate also how to use `.option` file. You can find some initial tests in `pb/nanopb` but is not yet being used. You can regenerate it using the following command, but note that you need to have a `protoc` binary.

```
generator-bin/protoc --nanopb_out=. onnx.proto
```

Note that nanopb archieves a signifiant reduction of the '.c' and `.h` files. 69K/45K for non nanopb and 14/20KB for nanopb. So (69+45)/(14+20) thats 3 times less!

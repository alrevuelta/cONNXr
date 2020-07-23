# Onnx File Generator

- uses onnx schema files
- generates operator header files
- generates operator stubs
- generates operator sanity checks
- generates operator type resolver
- generates operator name resolver


## Usage CLI

```
$ python -m onnx_generator -h
usage: onnx_generator [-h] [-v] [--header <path>] [--src <path>] [-i <regex> [<regex> ...]]
                      [-e <regex> [<regex> ...]]
                      [--] <path>

positional arguments:
  [--] <path>           where to put generated files

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose output
  --header <path>       where to put headers in main path (default: include/operators)
  --src <path>          where to put sources in main path (default: src/operators)
  -i <regex> [<regex> ...], --include <regex> [<regex> ...]
                        operators to include (default: ['.*'])
  -e <regex> [<regex> ...], --exclude <regex> [<regex> ...]
                        already included operators to exclude (default: [])
```

## Usage python

```python
import onnx_generator
import onnx_generator.args as args
print(args.path)
### []
print(args.include)
### ['.*']
print(args.exclude)
### []
print(args.src)
### ['src/operators']
print(args.header)
### ['include/operators']
print(args.verbose)
### 0
import onnx_generator.run as run
### including onnx operator schemas
### pattern '.*' included 324 operator schemas
### result: 324 onnx operator schemas
### excluding onnx operator schemas
### result: 324 onnx operator schemas
### generating onnx operator headers
### skipping write because args.path not set
### generating onnx operator implementations
### skipping write because args.path not set
len(run.schemas)
### 324
len(run.headers)
### 324
len(run.implementations)
### 324
```

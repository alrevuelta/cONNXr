import argparse
import os
import sys
import re
from .OperatorHeader import OperatorHeader
from .OperatorImplementation import OperatorImplementation
from .OnnxWrapper import OnnxSchema

parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx",
    nargs=1,
    metavar="<path>",
    help="use custom onnx_cpp2py_export.so located in <path> dir"
)
parser.add_argument(
    "-v", "--verbose",
    action='count', default=0,
    help="verbose output"
)
parser.add_argument(
    "--header",
    nargs=1,
    metavar="<path>",
    default=["include/operators"],
    help="where to put headers in main path"
)
parser.add_argument(
    "--src",
    nargs=1,
    metavar="<path>",
    default=["src/operators"],
    help="where to put sources in main path"
)
parser.add_argument(
    "-i", "--include",
    metavar="<regex>",
    nargs='+',
    default=[".*"],
    help="operators to include (default: '.*')"
)
parser.add_argument(
    "-e", "--exclude",
    nargs='+',
    metavar="<regex>",
    help="already included operators to exclude (default: None)"
)
parser.add_argument(
    'path',
    metavar="[--] <path>",
    nargs=1,
    help="where to put generated files"
)
args = parser.parse_args()

print(args)

if args.onnx:
    sys.path.insert(0, os.path.realpath(args.onnx[0]))
    import onnx_cpp2py_export
else:
    from onnx import onnx_cpp2py_export

all_schemas = [ OnnxSchema(s) for s in onnx_cpp2py_export.defs.get_all_schemas_with_history()]

schemas = []
for pattern in args.include:
  included = list(filter(lambda s: re.match(pattern,s.operator_name) != None,all_schemas))
  for s in included:
    if (args.verbose):
      print(f"pattern '{pattern}' included '{s.operator_name}'")
    if s not in schemas:
      schemas.append(s)
  print(f"pattern '{pattern}' included {len(included)} operators")


if args.exclude:
  for pattern in args.exclude:
    excluded = list(filter(lambda s: re.match(pattern,s.operator_name) != None ,schemas))
    for s in excluded:
      if args.verbose:
        print(f"pattern '{pattern}' excluded '{s.operator_name}'")
      schemas.remove(s)
  print(f"pattern '{pattern}' excluded {len(excluded)} operators")

headers = [ OperatorHeader(s,args.header[0]) for s in schemas ]

for h in headers:
  filename = os.path.normpath(f"{h.schema.operator_name}.h")
  dirname = os.path.realpath(f"{args.path[0]}/{args.header[0]}/{h.schema.domain}/")
  filepath = os.path.realpath(f"{dirname}/{filename}")
  if args.verbose:
    print(f"writing header {filepath}")
  os.makedirs(dirname,exist_ok=True)
  open(filepath,"w").write(h.text())
print(f"wrote {len(headers)} headers")

implementations = [ OperatorImplementation(s,args.header[0]) for s in schemas ]

for imp in implementations:
  filename = os.path.normpath(f"{imp.schema.operator_name}.c")
  dirname = os.path.realpath(f"{args.path[0]}/{args.src[0]}/{imp.schema.domain}/")
  filepath = os.path.realpath(f"{dirname}/{filename}")
  if args.verbose:
    print(f"writing source {filepath}")
  os.makedirs(dirname,exist_ok=True)
  open(filepath,"w").write(imp.text())
print(f"wrote {len(headers)} source")
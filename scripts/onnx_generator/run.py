import os
import sys
import re
from .OperatorHeader import OperatorHeader
from .OperatorImplementation import OperatorImplementation
from .OnnxWrapper import OnnxSchema
from . import args

if args.onnx:
    sys.path.insert(0, os.path.realpath(args.onnx[0]))
    import onnx_cpp2py_export
else:
    from onnx import onnx_cpp2py_export

all_schemas = [ OnnxSchema(s) for s in onnx_cpp2py_export.defs.get_all_schemas_with_history()]

schemas = []
print("including onnx operator schemas")
for pattern in args.include:
    included = list(filter(lambda s: re.match(pattern,s.operator_name) != None,all_schemas))
    for s in included:
        if (args.verbose):
            print(f"pattern '{pattern}' included '{s.operator_name}'")
        if s not in schemas:
            schemas.append(s)
    print(f"pattern '{pattern}' included {len(included)} operator schemas")
print(f"result: {len(schemas)} onnx operator schemas")

print("excluding onnx operator schemas")
for pattern in args.exclude:
    excluded = list(filter(lambda s: re.match(pattern,s.operator_name) != None ,schemas))
    for s in excluded:
        if args.verbose:
            print(f"pattern '{pattern}' excluded '{s.operator_name}'")
        schemas.remove(s)
    print(f"pattern '{pattern}' excluded {len(excluded)} operators")
print(f"result: {len(schemas)} onnx operator schemas")

print("generating onnx operator headers")
headers = [ OperatorHeader(s,args.header[0]) for s in schemas ]

if args.path:
    for h in headers:
        filename = os.path.normpath(f"{h.schema.operator_name}.h")
        dirname = os.path.realpath(f"{args.path[0]}/{args.header[0]}/{h.schema.domain}/")
        filepath = os.path.realpath(f"{dirname}/{filename}")
        if args.verbose:
            print(f"writing header {filepath}")
        os.makedirs(dirname,exist_ok=True)
        open(filepath,"w").write(h.text())
    print(f"wrote {len(headers)} headers")
else:
    print("skipping write because args.path not set")

print("generating onnx operator implementations")
implementations = [ OperatorImplementation(s,args.header[0]) for s in schemas ]

if args.path:
    for imp in implementations:
        filename = os.path.normpath(f"{imp.schema.operator_name}.c")
        dirname = os.path.realpath(f"{args.path[0]}/{args.src[0]}/{imp.schema.domain}/")
        filepath = os.path.realpath(f"{dirname}/{filename}")
        if args.verbose:
            print(f"writing source {filepath}")
        os.makedirs(dirname,exist_ok=True)
        open(filepath,"w").write(imp.text())
    print(f"wrote {len(headers)} source")
else:
    print("skipping write because args.path not set")
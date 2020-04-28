import os
import sys
import re
import pathlib
from .OperatorHeader import OperatorHeader
from .OperatorTypeResolver import OperatorTypeResolver
from .OperatorSanityCheck import OperatorSanityCheck
from .OperatorSets import OperatorSets
from .OnnxWrapper import OnnxSchema
from . import args

def note(text, verbosity=0 ):
    if verbosity <= args.verbose:
        print(text)

def warning(text, verbosity=0):
    if verbosity <= args.verbose:
        print(text, file=sys.stderr)

def fatal(text, error=1):
    print(text, file=sys.stderr)
    sys.exit(error)

if args.onnx:
    sys.path.insert(0, os.path.realpath(args.onnx[0]))
    try:
        import onnx_cpp2py_export
    except:
        fatal("could not import onnx_cpp2py_export!")
else:
    try:
        from onnx import onnx_cpp2py_export
    except:
        fatal("could not import onnx_cpp2py_export from onnx!")

all_schemas = [ OnnxSchema(s) for s in onnx_cpp2py_export.defs.get_all_schemas_with_history()]

schemas = []
note("including onnx operator schemas",1)
for pattern in args.include:
    included = list(filter(lambda s: re.match(pattern,s.operator_name) != None,all_schemas))
    for s in included:
        note(f"pattern '{pattern}' included '{s.operator_name}'",3)
        if s not in schemas:
            schemas.append(s)
    note(f"pattern '{pattern}' included {len(included)} operator schemas",2)
note(f"continuing with {len(schemas)} of {len(all_schemas)} onnx operator schemas",1)

note("excluding onnx operator schemas",1)
for pattern in args.exclude:
    excluded = list(filter(lambda s: re.match(pattern,s.operator_name) != None ,schemas))
    for s in excluded:
        note(f"pattern '{pattern}' excluded '{s.operator_name}'",3)
        schemas.remove(s)
    note(f"pattern '{pattern}' excluded {len(excluded)} operators",2)
note(f"continuing with {len(schemas)} of {len(all_schemas)} onnx operator schemas",1)

note("generating onnx operator headers")
path = f"{args.path[0]}/{args.header[0]}/"
headers = [ OperatorHeader(s,path) for s in schemas ]
note("generating onnx operator type resolvers")
path = f"{args.path[0]}/{args.resolve[0]}/"
resolvers = [ OperatorTypeResolver(s,path) for s in schemas ]
note("generating onnx operator sanity checks")
path = f"{args.path[0]}/{args.check[0]}/"
checks = [ OperatorSanityCheck(s,path) for s in schemas ]
note("generating onnx operator sets")
path = f"{args.path[0]}/{args.sets[0]}/"
sets = OperatorSets(schemas,path)

files = []
if not args.path:
    warning("skipping write because args.path is not set")
else:
    if not args.no_header:
        for h in headers:
            files.append(h)
    if not args.no_resolve:
        for r in resolvers:
            files.append(r)
    if not args.no_check:
        for c in checks:
            files.append(c)
    if not args.no_sets:
        files.append(sets)

writecount = 0
note("Writing files",1)
for obj in files:
    path = obj.filename().resolve()
    if path.exists() and not args.force:
        warning(f"skipping existing file '{path}'",1)
        continue
    note(f"writing file {path}",3)
    os.makedirs(path.parent,exist_ok=True)
    path.open("w").write(obj.text())
    writecount += 1
note(f"wrote {writecount} of {len(files)} files")

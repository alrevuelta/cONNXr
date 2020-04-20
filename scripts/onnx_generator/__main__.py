import argparse
from . import args

parser = argparse.ArgumentParser("onnx_generator")
parser.add_argument(
    "--onnx",
    nargs=1,
    metavar="<path>",
    default = args.onnx,
    help="use custom onnx_cpp2py_export.so located in <path> dir"
)
parser.add_argument(
    "-v", "--verbose",
    action='count',
    default=args.verbose,
    help="verbose output"
)
parser.add_argument(
    "--header",
    nargs=1,
    metavar="<path>",
    default=args.header,
    help=f"where to put headers in main path (default: {args.header[0]})"
)
parser.add_argument(
    "--src",
    nargs=1,
    metavar="<path>",
    default=args.src,
    help=f"where to put sources in main path (default: {args.src[0]})"
)
parser.add_argument(
    "-i", "--include",
    metavar="<regex>",
    nargs='+',
    default=args.include,
    help=f"operators to include (default: {args.include})"
)
parser.add_argument(
    "-e", "--exclude",
    nargs='+',
    metavar="<regex>",
    default=args.exclude,
    help=f"already included operators to exclude (default: {args.exclude})"
)
parser.add_argument(
    'path',
    metavar="[--] <path>",
    default=args.path,
    nargs=1,
    help="where to put generated files"
)

args.__dict__.update(parser.parse_args().__dict__)

import run

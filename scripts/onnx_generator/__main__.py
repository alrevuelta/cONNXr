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
    "-f", "--force",
    action='count',
    default=args.force,
    help="overwrite existing files"
)
parser.add_argument(
    "--no-header",
    action='count',
    default=args.no_header,
    help="don't write headers"
)
parser.add_argument(
    "--header",
    nargs=1,
    metavar="<path>",
    default=args.header,
    help=f"where to put headers in main path (default: {args.header[0]})"
)
parser.add_argument(
    "--no-resolve",
    action='count',
    default=args.no_resolve,
    help="don't write resolver src"
)
parser.add_argument(
    "--resolve",
    nargs=1,
    metavar="<path>",
    default=args.resolve,
    help=f"where to put resolver src in main path (default: {args.resolve[0]})"
)
parser.add_argument(
    "--no-check",
    action='count',
    default=args.no_check,
    help="don't write check src"
)
parser.add_argument(
    "--check",
    nargs=1,
    metavar="<path>",
    default=args.check,
    help=f"where to put check src in main path (default: {args.check[0]})"
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

from . import run

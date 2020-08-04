import argparse
from . import args

parser = argparse.ArgumentParser("onnx_generator")
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
    "--force-header",
    action='count',
    default=args.force_header,
    help="overwrite headers"
)
parser.add_argument(
    "--force-resolve",
    action='count',
    default=args.force_resolve,
    help="overwrite resolver src"
)
parser.add_argument(
    "--force-sets",
    action='count',
    default=args.force_sets,
    help="overwrite sets src"
)
parser.add_argument(
    "--force-template",
    action='count',
    default=args.force_template,
    help="overwrite template src"
)
parser.add_argument(
    "--force-info",
    action='count',
    default=args.force_info,
    help="overwrite info src"
)
parser.add_argument(
    "-n", "--dryrun",
    action='count',
    default=args.dryrun,
    help="do not write anything"
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
    "--no-sets",
    action='count',
    default=args.no_sets,
    help="don't write sets src"
)
parser.add_argument(
    "--sets",
    nargs=1,
    metavar="<path>",
    default=args.sets,
    help=f"where to put sets src in main path (default: {args.sets[0]})"
)
parser.add_argument(
    "--no-template",
    action='count',
    default=args.no_template,
    help="don't write template src"
)
parser.add_argument(
    "--template",
    nargs=1,
    metavar="<path>",
    default=args.template,
    help=f"where to put template src in main path (default: {args.template[0]})"
)
parser.add_argument(
    "--no-info",
    action='count',
    default=args.no_info,
    help="don't write info src"
)
parser.add_argument(
    "--info",
    nargs=1,
    metavar="<path>",
    default=args.info,
    help=f"where to put info src in main path (default: {args.info[0]})"
)
parser.add_argument(
    "--domains",
    nargs='+',
    metavar="<domain>",
    default=args.domains,
    help=f"what domain to include (default: {args.domains[0]})"
)
parser.add_argument(
    "--version",
    nargs=1,
    metavar="<version>",
    default=args.version,
    help=f"what version to include (<number>, 'latest' or 'all') (default: {args.version[0]})"
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

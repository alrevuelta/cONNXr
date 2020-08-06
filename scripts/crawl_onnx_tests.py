import argparse
import subprocess
import sys
import os
import pkg_resources
import glob

def parse_args():  # type: () -> argparse.Namespace
    """
    parser = argparse.ArgumentParser('backend-test-tools')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('generate-data', help='convert testcases to test data')
    subparser.add_argument('-o', '--output', default=DATA_DIR,
                           help='output directory (default: %(default)s)')
    subparser.set_defaults(func=generate_data)

    return parser.parse_args()
    """


# Hardcoded. Replace once #2918 is merged
# This tabled can be accessed programatically
VERSION_TABLE = [
    # Release-version, IR version, ai.onnx version, ai.onnx.ml version, (optional) ai.onnx.training version
    ('1.0', 3, 1, 1),
    ('1.1', 3, 5, 1),
    ('1.1.2', 3, 6, 1),
    #('1.2', 3, 7, 1), # TODO op 7 is missing. Is the table right? There is relase that matches 1.2
    ('1.3', 3, 8, 1),
    ('1.4.1', 4, 9, 1),
    ('1.5.0', 5, 10, 1),
    ('1.6.0', 6, 11, 2),
    ('1.7.0', 7, 12, 2, 1)
]

TEMP_DIR = "older_versions"
OUTPUT_DIR = "test_data/onnx_backend"
PYTHON_TESTS = "onnx/backend/test/data/node/"

def download(onnx_version):
    os.system(f"pip download --no-binary=:all: --no-deps -d {TEMP_DIR}/{onnx_version} onnx=={onnx_version}")

def main():  # type: () -> None
    #args = parse_args()
    #args.func(args)
    for version in VERSION_TABLE:
        download(version[0])

        tar_file_path = glob.glob(f'{TEMP_DIR}/{version[0]}/onnx-{version[0]}*.tar.gz')[0]
        folder_name = tar_file_path.replace(".tar.gz", "")
        tests_path = folder_name + "/" + PYTHON_TESTS

        os.system(f"tar -zxvf {tar_file_path} -C {TEMP_DIR}/{version[0]}")

        # TODO I would say that the tests in node folder only belong to ai.onnx domain
        out_folder = os.path.join(OUTPUT_DIR, f"node_ai.onnx.opset{version[2]}_release{version[0]}")
        os.system(f"mkdir {out_folder}")
        os.system(f"cp -R {tests_path} {out_folder}")

    
    os.system(f"rm -r {TEMP_DIR}")


if __name__ == '__main__':
    main()

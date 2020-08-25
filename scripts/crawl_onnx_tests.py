import argparse
import subprocess
import sys
import os
import pkg_resources
import glob
import onnx

def parse_args():  # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', help='Set for a specific release. By default all releases are crawled. Must match a release existing on PyPI')
    parser.add_argument('--domain', help='Set to fetch only a specific domain ai.onnx|ai.onnx.ml|ai.onnx.preview.training')
    parser.add_argument('--folder', default='test_data/onnx_backend', help='Output folder to place the tests')
    return parser.parse_args()


# This table can be accessed programatically (see onnx #2918)
# However I suspect there is a bug with the '1.2' entry, since such version
# does not exist
VERSION_TABLE = [
    # Release-version, IR version, ai.onnx version, ai.onnx.ml version, (optional) ai.onnx.training version
    ('1.0', 3, 1, 1),
    ('1.1', 3, 5, 1),
    ('1.1.2', 3, 6, 1),
    ('1.2.3', 3, 7, 1), # The original table is 1.2 but that release does not exist
    ('1.3', 3, 8, 1),
    ('1.4.1', 4, 9, 1),
    ('1.5.0', 5, 10, 1),
    ('1.6.0', 6, 11, 2),
    ('1.7.0', 7, 12, 2, 1)
]

TEMP_DIR = "temp"
PYTHON_TESTS = "onnx/backend/test/data/node/"

def download(onnx_version):
    os.system(f"pip download --no-binary=:all: --no-deps -d {TEMP_DIR}/{onnx_version} onnx=={onnx_version}")

def main():  # type: () -> None
    args = parse_args()
    if not args.release:
        releases = [i[0] for i in VERSION_TABLE]
    else:
        releases = [args.release]
        print(releases)

    # Iterate the PyPI releases
    for release in releases:
        # Download the Python release
        download(release)

        # Untar the files
        tar_file_path = glob.glob(f'{TEMP_DIR}/{release}/onnx-{release}*.tar.gz')[0]
        folder_name = tar_file_path.replace(".tar.gz", "")
        os.system(f"tar -zxvf {tar_file_path} -C {TEMP_DIR}/{release}")

        # Path to the tests folder containing all the onnx and pb files
        tests_path = os.path.join(folder_name, PYTHON_TESTS)        

        # Iterate test cases of a given release
        for test_folder in os.listdir(tests_path):
            test_folder_full = os.path.join(tests_path, test_folder)
            if os.path.isdir(test_folder_full):

                # Check if the test contains a model file. Note that onnx is used
                # in the last releases but .pb is used in the first ones
                included_extensions = ['model.onnx', 'node.pb']
                file_names = [fn for fn in os.listdir(test_folder_full) if any(fn.endswith(ext) for ext in included_extensions)]

                if len(file_names) != 1:
                    raise Exception("Model file not found")

                # Open the onnx file
                onnx_path = os.path.join(tests_path, test_folder, file_names[0])
                onnx_model = onnx.load(onnx_path)

                if len(onnx_model.opset_import) != 1:
                    # TODO v1.0 and v1.1 don't use onnx but pb. Seems that this field
                    # is empty
                    raise Exception("Opset imported different than one")

                # Check which opset is imported
                opset_version = onnx_model.opset_import[0].version
                domain_name = onnx_model.opset_import[0].domain

                # Base domain ai.onnx is empty
                if domain_name == "":
                    domain_name = "ai.onnx"

                # Just skip if a specific domain is provided
                if args.domain and args.domain != domain_name:
                    continue

                # Copy the files
                # Output folder
                output_path = os.path.join(args.folder,
                                        domain_name,
                                        str(opset_version),
                                        test_folder)                

                # Folder for domain
                domain_path = os.path.join(args.folder,
                                        domain_name)

                # Folder for opset 
                opset_path = os.path.join(args.folder,
                                        domain_name,
                                        str(opset_version))

                os.system(f"mkdir {domain_path}")
                os.system(f"mkdir {opset_path}")
                os.system(f"cp -R {test_folder_full} {output_path}")

    #Commented for testing
    #os.system(f"rm -r {TEMP_DIR}")

if __name__ == '__main__':
    main()

import argparse
import subprocess
import sys
import os
import pkg_resources
import glob
import onnx

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
    #('1.0', 3, 1, 1),
    #('1.1', 3, 5, 1),
    #('1.1.2', 3, 6, 1),
    #('1.2', 3, 7, 1), # TODO op 7 is missing. Is the table right? There is relase that matches 1.2
    ('1.3', 3, 8, 1),
    ('1.4.1', 4, 9, 1),
    ('1.5.0', 5, 10, 1),
    ('1.6.0', 6, 11, 2),
    #('1.7.0', 7, 12, 2, 1)
]

TEMP_DIR = "older_versions"
#OUTPUT_DIR = "test_data/onnx_backend"
OUTPUT_DIR = "remove"
PYTHON_TESTS = "onnx/backend/test/data/node/"

map_domain_name = {"": "ai.onnx",
                  "xx": "ai.onnx.ml",
                  "xxx": "ai.onnx.training"}

def download(onnx_version):
    os.system(f"pip download --no-binary=:all: --no-deps -d {TEMP_DIR}/{onnx_version} onnx=={onnx_version}")

def main():  # type: () -> None
    #args = parse_args()
    #args.func(args)
    for version in VERSION_TABLE:
        # Download the Python release
        download(version[0])

        # Untar the files in a temp dir
        tar_file_path = glob.glob(f'{TEMP_DIR}/{version[0]}/onnx-{version[0]}*.tar.gz')[0]
        folder_name = tar_file_path.replace(".tar.gz", "")
        os.system(f"tar -zxvf {tar_file_path} -C {TEMP_DIR}/{version[0]}")

        tests_path = os.path.join(folder_name, PYTHON_TESTS)        

        # Iterate test cases of a given release
        for test_folder in os.listdir(tests_path):
            test_folder_full = os.path.join(tests_path, test_folder)
            if os.path.isdir(test_folder_full):

                # Check if the test contains a model file. Note that onnx is used
                # in the last releases but .pb is used in the first ones
                included_extensions = ['model.onnx','node.pb']
                file_names = [fn for fn in os.listdir(test_folder_full) if any(fn.endswith(ext) for ext in included_extensions)]

                if len(file_names) != 1:
                    raise Exception("Model file not found")

                # Open the onnx file
                onnx_path = os.path.join(tests_path, test_folder, file_names[0])
                onnx_model = onnx.load(onnx_path)

                if len(onnx_model.opset_import) != 1:
                    print(dir(onnx_model))
                    print(onnx_model.model_version)
                    print(onnx_model.opset_import)
                    raise Exception("Opset imported different than one")

                # Check which opset is imported
                opset_version = onnx_model.opset_import[0].version

                # Ouput folder
                output_path = os.path.join(OUTPUT_DIR,
                                           map_domain_name[onnx_model.domain],
                                           str(opset_version),
                                           test_folder)                

                # Folder for domain
                domain_path = os.path.join(OUTPUT_DIR,
                                           map_domain_name[onnx_model.domain])

                # Folder for opset 
                opset_path = os.path.join(OUTPUT_DIR,
                                          map_domain_name[onnx_model.domain],
                                          str(opset_version))

                print("creating folder domain", domain_path)
                print("creaging folder, opset", opset_path)
                print("copy", test_folder_full, "to", output_path)
                os.system(f"mkdir {domain_path}")
                os.system(f"mkdir {opset_path}")
                os.system(f"cp -R {test_folder_full} {output_path}")

    #Commented for testing
    #os.system(f"rm -r {TEMP_DIR}")


if __name__ == '__main__':
    main()

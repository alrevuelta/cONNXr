from ctypes import *

connxr = CDLL('build/connxr.so')

SAVED_TIMES = {}

# TODO Run benchmarking without any tracing active.
# See makefile
def benchmark_model(model_id, model_path, io_path, n_inputs, n_outputs, n_runs=1):
    test_model_function = connxr.test_model
    test_model_function.restype = c_double

    model_times = []

    # Run n_runs iterations of inference to average
    for i in range(n_runs):
        # Reusing the test function.
        result = test_model_function(model_id, model_path, io_path, n_inputs, n_outputs)
        if result > 0:
            model_times.append(result)

        else:
            raise Exception("The output of the model doesn't match the expected")

    SAVED_TIMES[model_id] = model_times

if __name__ == "__main__":
    print("Benchmarking models:")

    benchmark_model(b"mnist",
                    b"test/mnist/model.onnx",
                    b"test/mnist/test_data_set_0",
                    1, 1, n_runs=10)

    benchmark_model(b"mobilenetv2",
                    b"test/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
                    b"test/mobilenetv2-1.0/test_data_set_0",
                    1, 1, n_runs=1)

    benchmark_model(b"super_resolution",
                    b"test/super_resolution/super_resolution.onnx",
                    b"test/super_resolution/test_data_set_0",
                    1, 1, n_runs=1)

    benchmark_model(b"tinyyolov2",
                    b"test/tiny_yolov2/Model.onnx",
                    b"test/tiny_yolov2/test_data_set_0",
                    1, 1, n_runs=1)

    # Print report at the end
    print("-------------------------------")
    for model, times in SAVED_TIMES.items():
        avg = sum(times)/len(times)
        print("Model:", str(model), "| Average Time:", avg, "s", "| All runs:", times)

from ctypes import *

connxr = cdll.LoadLibrary('build/libconnxr.so')

test_model_function = connxr.test_model
test_model_function.restype = c_double

SAVED_TIMES = {}

# TODO Run benchmarking without any tracing active.
# See makefile
def benchmark_model(id, path, io, n_inputs, n_outputs, n_runs=1):
    model_times = []

    # Run n_runs iterations of inference to average
    for i in range(n_runs):
        # Reusing the test function.
        print("Running", id)
        result = test_model_function(id.encode('UTF-8'),
                                     path.encode('UTF-8'),
                                     io.encode('UTF-8'),
                                     n_inputs, n_outputs)
        if result > 0:
            model_times.append(result)
        elif result == 0:
            print("[Warning] Measured time is 0. Perhaps trying to measure a very small time."+
                  "There is a known issue with time library in Windows.")
            model_times.append(result)
        else:
            raise Exception("The output of the model doesn't match the expected")

    SAVED_TIMES[id] = model_times

if __name__ == "__main__":
    print("Benchmarking models:")

    benchmark_model("mnist",
                    "test/mnist/model.onnx",
                    "test/mnist/test_data_set_0",
                    1, 1, n_runs=10)

    benchmark_model("mobilenetv2",
                    "test/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
                    "test/mobilenetv2-1.0/test_data_set_0",
                    1, 1, n_runs=1)

    benchmark_model("super_resolution",
                    "test/super_resolution/super_resolution.onnx",
                    "test/super_resolution/test_data_set_0",
                    1, 1, n_runs=1)

    benchmark_model("tinyyolov2",
                    "test/tiny_yolov2/Model.onnx",
                    "test/tiny_yolov2/test_data_set_0",
                    1, 1, n_runs=1)

    # Print report at the end
    print("-------------------------------")
    for model, times in SAVED_TIMES.items():
        avg = sum(times)/len(times)
        print("Model:", str(model), "| Average Time:", avg, "s", "| All runs:", times)

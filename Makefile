all: clean build
	./runtest $(ts) $(tc)

clean:
	echo "Cleaning"
	rm -f runtest
	rm -f connxr

build:
	echo "Building"
	gcc -std=c99 -Wall -D TRACE_LEVEL=0 -o runtest test/tests.c src/operators/*.c src/inference.c src/utils.c src/trace.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lcunit

onnx_backend_tests:
	echo "Running onnx backend tests"
	./runtest onnxBackendSuite

onnx_models_tests:
	echo "Running models tests"
	./runtest modelsTestSuite

# TODO Benchmarking should run without many logging crap to avoid performance loss
# All runs will be average later on in the post processing phase
benchmark: clean build
	echo "Runing benchmarking"

	# Run 10 iterations for mnist to average
	number=1 ; while [[ $$number -le 10 ]] ; do \
		echo "Benchmarking iteration "$$number ; \
		./runtest modelsTestSuite test_model_mnist >> benchmarking.txt ; \
		((number = number + 1)) ; \
  done

	# Run only 1 iteration of tinyyolo (it takes a lot to run)
	./runtest modelsTestSuite test_model_tinyyolov2 >> benchmarking.txt

	# Run some postprocessing on the benchmarking results
	python3 scripts/parse_output_benchmarking.py

valgrind:
	echo "TODO: Running valgrind"
	#rm -f runprofile
	#rm -f call*
	#gcc -std=c99 -Wall -D xxx -o runprofile test/tests.c src/operators/*.c src/*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lcunit
	#valgrind --tool=callgrind ./runprofile $(ts) $(tc)
	#qcachegrind

make build_cli:
	rm -f connxr
	gcc -std=c99 -Wall -D TRACE_LEVEL=0 -o connxr src/operators/*.c src/*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lm

#memory leak stuff TODO:

#nanopb:
#	rm -f prototest
#	gcc -std=c99 -Wall -D xxx -o prototest test_nanopb.c ../src/pb/nanopb/onnx.pb.c ../src/pb/nanopb/pb_common.c ../src/pb/nanopb/pb_decode.c ../src/pb/nanopb/pb_encode.c -I/usr/local/include -L/usr/local/lib -lcunit
#	./prototest $(ts) $(tc)

#gprof:
#	rm -f gprof
#	gcc -std=c99 -D xxx -pg ../src/operators/*.c ../src/trace.c ../src/utils.c ../src/inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
#	./gprof $(ts) $(tc)

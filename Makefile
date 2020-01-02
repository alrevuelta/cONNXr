all: clean build
	./runtest $(ts) $(tc)

clean:
	echo "Cleaning"
	rm -f runtest
	rm -f connxr

build:
	echo "Building"
	# TODO: Do not include connxr.c because it contains another main. /e*.c is a quick fix
	gcc -std=c99 -Wall -D TRACE_LEVEL=0 -o runtest test/tests.c src/operators/*.c src/e*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lcunit

onnx_backend_tests:
	echo "Running onnx backend tests"
	./runtest onnxBackendSuite

onnx_models_tests:
	echo "Running models tests"
	./runtest modelsTestSuite

# TODO Benchmarking should run without many logging crap to avoid performance loss
# All runs will be average later on in the post processing phase
benchmark:
	echo "Runing benchmarking"
	number=1 ; while [[ $$number -le 10 ]] ; do \
		echo "Benchmarking iteration "$$number ; \
		./runtest modelsTestSuite test_model_mnist >> benchmarking.txt ; \
		((number = number + 1)) ; \
  done

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
	# TODO Set different levels of verbosity
	gcc -std=c99 -Wall -o connxr src/operators/*.c src/*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c

#memory leak stuff TODO:

#nanopb:
#	rm -f prototest
#	gcc -std=c99 -Wall -D xxx -o prototest test_nanopb.c ../src/pb/nanopb/onnx.pb.c ../src/pb/nanopb/pb_common.c ../src/pb/nanopb/pb_decode.c ../src/pb/nanopb/pb_encode.c -I/usr/local/include -L/usr/local/lib -lcunit
#	./prototest $(ts) $(tc)

#gprof:
#	rm -f gprof
#	gcc -std=c99 -D xxx -pg ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
#	./gprof $(ts) $(tc)

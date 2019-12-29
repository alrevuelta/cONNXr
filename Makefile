all: clean build
	./runtest $(ts) $(tc)

clean:
	echo "Cleaning"
	rm -f runtest

build:
	echo "Building"
	gcc -std=c99 -Wall -D DEBUG -o runtest test/tests.c src/operators/*.c src/*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lcunit

onnx_backend_tests:
	echo "Running onnx backend tests"
	./runtest onnxBackendSuite

onnx_models_tests:
	echo "Running models tests"
	./runtest modelsTestSuite

benchmark:
	echo "TODO: Runing benchmarking"

valgrind:
	echo "TODO: Running valgrind"
	#rm -f runprofile
	#rm -f call*
	#gcc -std=c99 -Wall -D DEBUG -o runprofile test/tests.c src/operators/*.c src/*.c src/pb/onnx.pb-c.c src/pb/protobuf-c.c -lcunit
	#valgrind --tool=callgrind ./runprofile $(ts) $(tc)
	#qcachegrind

#memory leak stuff TODO:

#nanopb:
#	rm -f prototest
#	gcc -std=c99 -Wall -D DEBUG -o prototest test_nanopb.c ../src/pb/nanopb/onnx.pb.c ../src/pb/nanopb/pb_common.c ../src/pb/nanopb/pb_decode.c ../src/pb/nanopb/pb_encode.c -I/usr/local/include -L/usr/local/lib -lcunit
#	./prototest $(ts) $(tc)

#gprof:
#	rm -f gprof
#	gcc -std=c99 -D DEBUG -pg ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
#	./gprof $(ts) $(tc)

test:
	rm -f runtest
	gcc -std=c99 -Wall -D DEBUG -o runtest tests.c ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	./runtest $(ts) $(tc)

profile:
	rm -f runprofile
	rm -f call*
	gcc -std=c99 -Wall -D DEBUG -o runprofile tests.c ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	valgrind --tool=callgrind ./runprofile $(ts) $(tc)
	qcachegrind

nanopb:
	rm -f prototest
	gcc -std=c99 -Wall -D DEBUG -o prototest test_nanopb.c ../src/pb/nanopb/onnx.pb.c ../src/pb/nanopb/pb_common.c ../src/pb/nanopb/pb_decode.c ../src/pb/nanopb/pb_encode.c -I/usr/local/include -L/usr/local/lib -lcunit
	./prototest $(ts) $(tc)

gprof:
	rm -f gprof
	gcc -std=c99 -D DEBUG -pg ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -o gprof tests.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	./gprof $(ts) $(tc)

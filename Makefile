all:
	gcc -std=c99 -Wall -D DEBUG -o runtest test/tests.c src/operators/*.c src/embeddedml_debug.c src/embeddedml_utils.c src/embeddedml_inference.c src/pb/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	./runtest $(ts) $(tc)

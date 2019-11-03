test: tests.c
	gcc -D DEBUG -o test tests.c ../src/embeddedml_debug.c ../src/embeddedml_inference.c ../src/embeddedml_operators.c ../src/embeddedml_utils.c ../src/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c

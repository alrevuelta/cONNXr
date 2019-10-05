main: main.c
	gcc -o main main.c embeddedml_debug.c embeddedml_operators.c embeddedml_utils.c embeddedml_opwrapper.c embeddedml_inference.c onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lprotobuf-c
test: tests.c
	gcc -o test tests.c -I/usr/local/include -L/usr/local/lib -lcunit

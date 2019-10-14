main: main.c
	gcc -D DEBUG -o main main.c embeddedml_debug.c embeddedml_operators.c embeddedml_utils.c embeddedml_opwrapper.c embeddedml_inference.c onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lprotobuf-c

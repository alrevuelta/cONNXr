build:
	rm -f runtest
	gcc -std=c99 -Wall -D DEBUG -lm -O -c ../src/operators/*.c ../src/embeddedml_debug.c ../src/embeddedml_utils.c ../src/embeddedml_inference.c ../src/pb/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	ar rcs libonnxcruntimelib.a *.o
	#ranlib libonnxcruntimelib.a
	gcc -std=c99 -o example example.c libonnxcruntimelib.a
clean:
	rm -f *.o *.a *.gch

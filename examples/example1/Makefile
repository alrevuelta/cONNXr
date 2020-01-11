build:
	rm -f runtest
	gcc -std=c99 -Wall -D DEBUG -lm -O -c ../../src/operators/*.c ../../src/trace.c ../../src/utils.c ../../src/inference.c ../../src/pb/onnx.pb-c.c
	ar rcs libonnxcruntimelib.a *.o
	#ranlib libonnxcruntimelib.a
	gcc -std=c99 -o example example.c libonnxcruntimelib.a
clean:
	rm -f *.o *.a *.gch

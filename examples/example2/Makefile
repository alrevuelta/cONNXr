build:
	gcc -std=c99 -Wall -D TRACE_LEVEL=0 -o example2 example2.c ../../src/operators/*.c ../../src/inference.c ../../src/utils.c ../../src/trace.c ../../src/pb/onnx.pb-c.c ../../src/pb/protobuf-c.c
clean:
	rm -f *.o *.a *.gch
run:
	./example2

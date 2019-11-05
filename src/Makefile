main:
	gcc -std=c99 -c -Wall -D DEBUG operators/*.c embeddedml_debug.c embeddedml_utils.c embeddedml_inference.c onnx.pb-c.c
	mv *.o objdir
clean:
	rm -f objdir/*.o

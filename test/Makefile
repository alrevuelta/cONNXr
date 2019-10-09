test: tests.c
	gcc -o test tests.c ../src/embeddedml_operators.c ../src/embeddedml_opwrapper.c ../src/onnx.pb-c.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
prototype: prototype_python.c
	gcc -o prototype prototype_python.c -I/usr/include/python2.7 -lpython2.7 -I/usr/local/include -L/usr/local/lib -lcunit

all:
	gcc -std=c99 -Wall -D DEBUG -o runtest test/tests.c src/*.c -I/usr/local/include -L/usr/local/lib -lcunit -lprotobuf-c
	./runtest $(ts) $(tc)

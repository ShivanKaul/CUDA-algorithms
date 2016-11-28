UNAME := $(shell uname)
CC = nvcc 
GCC = gcc
ifeq ($(UNAME), Darwin)
	GCC = gcc-6
endif

all: rectify pool convolve grid_4_4 grid_512_512

rectify: rectify.cu rectify.h
	$(CC) -o rectify rectify.cu lodepng.cpp 

pool: pool.cu
	$(CC) -o pool pool.cu lodepng.cpp

convolve: convolve.cu convolve.h
	$(CC) -o convolve convolve.cu lodepng.cpp 

grid_4_4: grid_4_4.cu
	$(CC) -o grid_4_4 grid_4_4.cu

grid_512_512: grid_512_512.cu grid_512_512.h
	$(CC) -o grid_512_512 grid_512_512.cu

dump: dump.c
	$(GCC) -o dump dump.c lodepng.c

testing: testing.c
	$(GCC) -o testing testing.c lodepng.c

test:
	$(GCC) -o test -std=c99 test_equality.c lodepng.c -lm

clean:
	-rm rectify -f
	-rm pool -f
	-rm convolve -f
	-rm shivan* -f
	-rm grid_4_4 -f
	-rm grid_512_512 -f
	-rm -r rectify.dSYM -f
	-rm test -f

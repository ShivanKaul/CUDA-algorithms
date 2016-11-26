UNAME := $(shell uname)
CC = nvcc 
ifeq ($(UNAME), Darwin)
	CC = gcc-6
endif

dump: dump.c
	$(CC) -o dump dump.c lodepng.c

pool: pool.cu
	$(CC) -o pool pool.cu lodepng.cpp

rectify: rectify.cu rectify.h
	nvcc -o rectify rectify.cu lodepng.cpp

convolve: convolve.c convolve.h
	$(CC) -g -o convolve convolve.c lodepng.c  -fopenmp

test:
	$(CC) -o test -std=c99 test_equality.c lodepng.c -lm

grid_4_4: grid_4_4.cu
	$(CC) -o grid_4_4 grid_4_4.cu

clean:
	-rm rectify -f
	-rm pool -f
	-rm convolve -f
	-rm shivan* -f
	-rm -r rectify.dSYM -f

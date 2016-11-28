#!/usr/bin/env bash

# Get person's name and number of iterations
if [ "$#" -ne 2 ]; then
    echo -e "Illegal number of parameters : please enter your name (just one word) and number of iterations"
    exit
fi

NAME=$1
ITERATIONS=$2

rm  results/*
echo -e "\nMaking test executable for Lab 1...\n"
make test
echo -e "\nMaking all executables...\n"
make
# Rectify
echo -e "\nRunning rectify and testing image...\n"
FILENAME=$NAME-rectify.png
./rectify test.png $FILENAME
./test test_rectify.png $FILENAME
mv $FILENAME results/
echo -e "Done! Saved at results/$FILENAME\n"

# Pool
echo -e "Running pool and testing image...\n"
FILENAME=$NAME-pool.png
./pool test.png $FILENAME
./test test_pool.png $FILENAME
mv $FILENAME results/
echo -e "Done! Saved at results/$FILENAME\n"

# Convolve
echo -e "Running convolve and testing image...\n"
FILENAME=$NAME-convolve.png
./convolve test.png $FILENAME
./test test_convolve.png $FILENAME
mv $FILENAME results/
echo -e "Done! Saved at results/$FILENAME\n"


# Grid 4x4
FILENAME=results/grid_4_4.txt
echo -e "Running grid_4_4 for $ITERATIONS iterations and storing results in $FILENAME...\n"
./grid_4_4 $ITERATIONS > $FILENAME

# Grid 512x512
FILENAME=results/grid_512_512.txt
echo -e "Running grid_512_512 for $ITERATIONS iterations and storing results in $FILENAME...\n"
./grid_512_512 $ITERATIONS > $FILENAME

echo -e "All done! Cleaning all executables\n"
make clean


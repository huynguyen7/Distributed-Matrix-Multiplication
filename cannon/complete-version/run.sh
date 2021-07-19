#!/bin/sh

export OMP_NUM_THREADS=8

#clang -L/opt/homebrew/lib -fopenmp naive_mat_mul.c -o main.out
#./main.out
#rm main.out

mpicc -L/opt/homebrew/lib -fopenmp cannon_mat_mul.c -o main.out
mpiexec --hostfile hostfile -np $1 main.out
rm main.out

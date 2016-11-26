#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include "gputimer.h"

#define BLOCK_WIDTH 16

__device__ void synthesize_inner(int i, int r, int c, double *u, double *u1, double *u2)
{
  double P = 0.5; 
  int N = 4;
  double E = 0.0002;
  if (r > 0 && r < N - 1 && c > 0 && c < N - 1)
  {
    u[i] = (P * (u1[i+N] + u1[i-N] + u1[i-1] + u1[i+1] - N * u1[i]) + 2 * u1[i] - (1 - E) * u2[i]) / (1 + E);
  }
}

__device__ void synthesize_edge(int i, int r, int c, double *u, double *u1, double *u2)
{
  double G = 0.75;
  int N = 4;
  if (r == 0)
  {
    u[i] = G * u[i+N];
  }
  else if (r == N - 1)
  {
    u[i] = G * u[i-N];
  }
  else if (c == 0)
  {
    u[i] = G * u[i+1];
  }
  else if (c == N - 1)
  {
    u[i] = G * u[i-1];
  }
}

__device__ void synthesize_corner(int i, int r, int c, double *u, double *u1, double *u2)
{
  double G = 0.75;
  int N = 4;
  if (r == 0 and c == 0)
  {
    u[i] = G * u[i+N];
  }
  else if (r == N - 1 and c == 0)
  {
    u[i] = G * u[i-N];
  }
  else if (r == 0 and c == N - 1)
  {
    u[i] = G * u[i-1];
  }
  else if (r == N - 1 and c == N - 1)
  {
    u[i] = G * u[i-1];
  }
}

__global__ void process(double *u, double *u1, double *u2, double *gr, int iterations)
{
    int tid = threadIdx.x;
    int r = tid/4;
    int c = tid%4;
    int i;
    for (i = 0; i < iterations; i++)
    {
        synthesize_inner(tid, r, c, u, u1, u2);
        __syncthreads();
        synthesize_edge(tid, r, c, u, u1, u2);
        __syncthreads();
        synthesize_corner(tid, r, c, u, u1, u2);
        __syncthreads();
        memcpy(u2, u1, 16 * sizeof(double));
        memcpy(u1, u, 16 * sizeof(double));
        gr[i] = u[10];
    }
}

int main(int argc, char *argv[])
{
  GpuTimer timer;
  if (argc < 2)
  {
    printf("Incorrect arguments! Input format: ./grid_4_4 <number of iteration> \n");
    return 1;
  }
  
  int i;
  int iterations = atoi(argv[1]);

  double u[16], u1[16], u2[16], r[iterations];
  double *gu, *gu1, *gu2, *gr;
  
  for (i = 0; i < 16; i++)
  {
    u[i] = 0;
    u1[i] = 0;
    u2[i] = 0;
  }
  u1[10] = 1;
  //printf("NUM_THREADS: %d, with width %d and height %d\n", NUM_THREADS, width, height);
  cudaMalloc(&gu, 16 * sizeof(double));
  cudaMalloc(&gu1, 16 * sizeof(double));
  cudaMalloc(&gu2, 16 * sizeof(double));
  cudaMalloc(&gr, iterations * sizeof(double));
  cudaMemcpy(gu, u, 16 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gu1, u1, 16 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gu2, u2, 16 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset((void *)gr, 0, iterations * sizeof(double));
  // launch the kernel
  timer.Start();
  process<<<1, BLOCK_WIDTH>>>(gu, gu1, gu2, gr, iterations);
  timer.Stop();
  // copy back the result array to the CPU
  cudaMemcpy(u, gu, 16 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(u1, gu1, 16 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(u2, gu2, 16 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r, gr, iterations * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(gu);
  cudaFree(gu1);
  cudaFree(gu2);
  cudaFree(gr);

/*  int j;
  for (i = 0; i < 4; i++)
  {
    for (j = 0; j < 4; j++)
    {
      printf(" %f", u[4*i + j]);
    }
    printf("\n");
  }
*/
  printf("%f", r[0]);
  for (i = 1; i < iterations; i++)
  {
    printf(", %f", r[i]);
  }

  printf("\nTime elapsed = %g ms with %d iterations\n", timer.Elapsed(), iterations);

//  free(u);
//  free(u1);
//  free(u2);
 
  return 0;
}

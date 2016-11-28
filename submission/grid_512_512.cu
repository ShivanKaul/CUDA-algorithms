#include "grid_512_512.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

__device__ void synthesize_inner(int r, float *u, float *u1, float *u2)
{
    if (r > 0 && r < N - 1)
    {
        int x, i;
        i = r * N;
        for (x = 1; x < N - 1; x++) {
            i++;
            u[i] = (P * (u1[i+N] + u1[i-N] + u1[i-1] + u1[i+1] - 4 * u1[i]) + 2 * u1[i] - (1 - E) * u2[i]) / (1 + E);
        }
    }
}

__device__ void synthesize_edge(int r, float *u)
{
    int i, x;
    i = r * N;
    if (r == 0)
    {
        for (x = 1; x < N-1; x++) {
            i++;
            u[i] = G * u[i+N];
        }
    }
    else if (r == N - 1)
    {
        for (x = 1; x < N-1; x++) {
            i++;
            u[i] = G * u[i-N];
        }
    } 
    else {
        u[i] = G * u[i+1];
        u[i + N - 1] = G * u[i + N - 2];
    }
}

__device__ void synthesize_corner(int r, float *u)
{
    if (r == 0)
    {
        u[0] = G * u[N];
        u[N-1] = G * u[N-2];
    }
    else if (r == N - 1)
    {
        int i = r * N;
        u[i] = G * u[i-N];
        u[i + N - 1] = G * u[i + N - 2];
    }
}

__global__ void process(float *u, float *u1, float *u2, float *result, int iterations, int center)
{
    int r = threadIdx.x; // One block : one row operated on by one thread
    int i;
    for (i = 0; i < iterations; i++)
    {
        synthesize_inner(r, u, u1, u2);
        __syncthreads();
        synthesize_edge(r, u);
        __syncthreads();
        synthesize_corner(r, u);
        __syncthreads();
        memcpy(u2, u1, N * N * sizeof(float));
        memcpy(u1, u, N * N * sizeof(float));
        result[i] = u[center];
    }
}

int main(int argc, char *argv[])
{
    GpuTimer timer;
    if (argc < 2)
    {
        printf("Incorrect arguments! Input format: ./grid_512_512 <number of iterations> \n");
        return 1;
    }

    int iterations = atoi(argv[1]);

    float h_result[iterations];
    float *d_u, *d_u1, *d_u2, *d_result;

    float h_u1[N * N] = {0}; // Set everything to 0
    

    int MIDDLE_INDEX = (N / 2) * N + (N / 2);

    // I'VE GOT BLISTERS ON MY FINGERS
    h_u1[MIDDLE_INDEX] = 1.0;

    cudaMalloc(&d_u, N * N * sizeof(float));
    checkCUDAError("Allocating memory to d_u on GPU");
    cudaMalloc(&d_u1, N * N * sizeof(float));
    checkCUDAError("Allocating memory to d_u1 on GPU");
    cudaMalloc(&d_u2, N * N * sizeof(float));
    checkCUDAError("Allocating memory to d_u2 on GPU");
    cudaMalloc(&d_result, iterations * sizeof(int));
    checkCUDAError("Allocating memory to d_result on GPU");
    // Special behaviour needed for u1 because of the drum hit
    cudaMemcpy(d_u1, h_u1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("Copying memory from h_u1 to d_u1 from host to device");
    // Set everything to 0
    cudaMemset((void *)d_u, 0, N * N * sizeof(float));
    cudaMemset((void *)d_u2, 0, N * N * sizeof(float));
    cudaMemset((void *)d_result, 0, iterations * sizeof(float));
    checkCUDAError("Memset d_u and d_u2 and d_result to be 0 on GPU");

    int TOTAL_THREADS = N;
    int THREADS_PER_BLOCK = N; // Each row is handled by a separate thread
    int BLOCKS_IN_GRID = (int)ceil( (float)TOTAL_THREADS / THREADS_PER_BLOCK);
    // launch the kernel
    timer.Start();
    process<<<BLOCKS_IN_GRID, THREADS_PER_BLOCK>>>(d_u, d_u1, d_u2, d_result, iterations, MIDDLE_INDEX);
    timer.Stop();
    checkCUDAError("Ran process on GPU");

    cudaMemcpy(h_result, d_result, iterations * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError("d_result -> h_result : device to host");
    
    int i;
    printf("%f", h_result[0]);
    for (i = 1; i < iterations; i++)
    {
        printf(", %f", h_result[i]);
    }

    printf("\nTime elapsed = %g ms with %d iterations\n", timer.Elapsed(), iterations);

    cudaFree(d_u);
    cudaFree(d_u1);
    cudaFree(d_u2);
    cudaFree(d_result);

    return 0;
}

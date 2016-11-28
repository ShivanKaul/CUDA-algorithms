#include "rectify.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

__global__ void getRect(unsigned char *image)
{
    // which thread is this?
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    image[idx] = (image[idx] < 127) ? 127 : image[idx];
}



void process(char *input_filename, char *output_filename)
{
    GpuTimer timer;

    unsigned error;
    unsigned char *image, *new_image;
    unsigned width, height;

    // Read in image
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error)
        printf("Error %u in lodepng: %s\n", error, lodepng_error_text(error));

    int TOTAL_PIXELS = width * height;
    int SIZE_OF_ARRAY = TOTAL_PIXELS * 4;

    cudaMalloc((void **) &new_image, SIZE_OF_ARRAY * sizeof(unsigned char));
    checkCUDAError("Allocating memory to new image on GPU");

    cudaMemcpy(new_image, image, SIZE_OF_ARRAY * sizeof(unsigned char), cudaMemcpyHostToDevice);
    checkCUDAError("Copying image to new image - host to device");

    int TOTAL_THREADS = TOTAL_PIXELS * 4;
    int THREADS_PER_BLOCK = 500;
    int BLOCKS_IN_GRID = (int)ceil( (float)TOTAL_THREADS / THREADS_PER_BLOCK);

    timer.Start();
    getRect<<<BLOCKS_IN_GRID, THREADS_PER_BLOCK>>>(new_image);
    timer.Stop(); 
    checkCUDAError("Check error in getRect");

    cudaMemcpy(image, new_image, SIZE_OF_ARRAY *sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // Check for any CUDA errors
    checkCUDAError("Copying new image to image - device to host");

    lodepng_encode32_file(output_filename, image, width, height);

    printf("Time elapsed = %g ms\n", timer.Elapsed());

    // Free GPU memory allocation and exit
    cudaFree(new_image);
    free(image);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Incorrect arguments! Input format: ./rectify <name of input png> <name of output png> < # threads> \n");
        return 1;
    }

    char *input_filename = argv[1];
    char *output_filename = argv[2];

    process(input_filename, output_filename);

    return 0;
}


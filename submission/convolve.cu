#include "convolve.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

__global__ void convolve(unsigned char *new_image, unsigned char *image, int width, int height, float *weights)
{
    // which thread is this?
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx + 1) % 4 != 0) {

        int i = idx / ((width - 2) * 4);
        int j = idx % ((width - 2) * 4);

        // convolveop
        float sum = 0.0;
        int rowPos, colPos;
        for (rowPos = 0; rowPos <= 2; rowPos++)
        {
            for (colPos = 0; colPos <= 2; colPos++)
            {
                int index = (width * 4 * (i + rowPos)) + j + 4 * colPos;
                sum += image[index] * weights[rowPos * 3 + colPos];
            }
        }

        int convolved = (int)sum;

        convolved = convolved > 255 ? 255 : convolved;
        convolved = convolved < 0 ? 0 : convolved;

        new_image[idx] = convolved;
    }
}



void process(char *input_filename, char *output_filename)
{
    GpuTimer timer;

    unsigned error;
    unsigned char *h_new_image, *d_new_image;
    unsigned char *h_image, *d_image;

    unsigned width, height; 
    float *d_w;
    float tempWeights[9];
    // To keep life easy, simply copy array elements
    // Could also have used mallocPitch
    int r, c;
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            tempWeights[r * 3 + c] = w[r][c];
        }
    }

    // Read in image
    error = lodepng_decode32_file(&h_image, &width, &height, input_filename);
    if (error)
        printf("Error %u in lodepng: %s\n", error, lodepng_error_text(error));
    int TOTAL_NEW_PIXELS = (width - 2) * (height - 2);
    int TOTAL_OLD_PIXELS = (width) * (height);
    int SIZE_OF_OLD_ARRAY = TOTAL_OLD_PIXELS * 4;
    int SIZE_OF_NEW_ARRAY = TOTAL_NEW_PIXELS * 4;

    h_new_image = (unsigned char *)malloc(SIZE_OF_NEW_ARRAY * sizeof(unsigned char));

    cudaMalloc((void **) &d_new_image, SIZE_OF_NEW_ARRAY * sizeof(unsigned char));
    checkCUDAError("Allocating memory to d new image on GPU");

    cudaMalloc((void **) &d_image, SIZE_OF_OLD_ARRAY * sizeof(unsigned char));
    checkCUDAError("Allocating memory to d old image on GPU");

    cudaMemcpy(d_image, h_image, SIZE_OF_OLD_ARRAY * sizeof(unsigned char), cudaMemcpyHostToDevice);
    checkCUDAError("Copying h image to d image - host to device");

    // Weights
    cudaMalloc((void **) &d_w, 9 * sizeof(float));
    checkCUDAError("Mallocing memory for weights on CUDA ");

    cudaMemcpy(d_w, tempWeights, 9 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("Copying weights to w - host to device");

    int TOTAL_THREADS = TOTAL_NEW_PIXELS * 4;
    int THREADS_PER_BLOCK = TOTAL_THREADS > 500 ? 500 : TOTAL_THREADS;
    int BLOCKS_IN_GRID = (int)ceil( (float)TOTAL_THREADS / THREADS_PER_BLOCK);

    timer.Start();
    convolve<<<BLOCKS_IN_GRID, THREADS_PER_BLOCK>>>(d_new_image, d_image, width, height, d_w);
    timer.Stop(); 
    checkCUDAError("Check error in convolve");

    cudaMemcpy(h_new_image, d_new_image, SIZE_OF_NEW_ARRAY *sizeof(unsigned char), cudaMemcpyDeviceToHost);
    checkCUDAError("Copying d_new_image to h_new_image - device to host");

    lodepng_encode32_file(output_filename, h_new_image, width-2, height-2);

    printf("Time elapsed = %g ms\n", timer.Elapsed());

    // Free GPU memory allocation and exit
    cudaFree(d_new_image);
    cudaFree(d_image);
    free(h_image);
    free(h_new_image);
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

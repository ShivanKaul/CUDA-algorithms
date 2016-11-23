#include "lodepng.h"
#include "gputimer.h"

#define MAX(a,b) ((a) > (b) ? a : b)

#define BLOCK_WIDTH 1000


__device__ int poolOp(unsigned char *i, int p, unsigned width)
{
  return MAX(i[p], MAX(i[p + 4], MAX(i[p + (width * 4)], i[p + (width * 4) + 4])));
}

__global__ void process(unsigned char *image,unsigned char *new_image, int NUM_THREADS, unsigned width, unsigned height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = (height/2 * width/2 * 4) / NUM_THREADS;
    int start_idx = tid * chunk_size;
    int end_idx = (tid == NUM_THREADS - 1) ? (height * width) : start_idx + chunk_size;
    int idx = start_idx;
    int pos = (2 * (idx / (width * 2)) * width * 4) + (2 * (idx%(width * 2))); 
    pos -= idx%4;
    if (idx%4 == 0 && tid != 0) {
        pos -= 4;
    }
    for (idx = start_idx; idx < end_idx; idx++)
    {
      if (idx > 0 && idx % 4 == 0) {
        pos += 4;
      }
      if (idx % (width * 2) == 0 && idx != 0 || (pos / (width * 4)) % 2 ==1) {
        pos += width * 4;
      }
      int ang = poolOp(image, pos, width);
      new_image[idx] = ang;
      pos++;
    }
  }
}

int main(int argc, char *argv[])
{
  GpuTimer timer;
  if (argc < 3)
  {
    printf("Incorrect arguments! Input format: ./pool <name of input png> <name of output png> \n");
    return;
  }
  
  char *input_filename = argv[1];
  char *output_filename = argv[2];

  unsigned error;
  unsigned char *image, *new_image, *gimage, *gnew_image;
  unsigned width, height;
  int NUM_THREADS;

  // Read in image
  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if (error)
    printf("Error %u in lodepng: %s\n", error, lodepng_error_text(error));

  NUM_THREADS = width * height;
  new_image = malloc(width * height * sizeof(unsigned char));
  cudaMalloc(&gimage, 4 * width * height * sizeof(unsigned char));
  cudaMalloc(&gnew_image, width * height * sizeof(unsigned char));
  cudaMemcpy(gimage, image, 4 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // launch the kernel
  timer.Start();
  process<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(gimage, gnew_image, NUM_THREADS, width, height);
  timer.Stop();
  // copy back the result array to the CPU
  cudaMemcpy(new_image, gnew_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(gnew_image);
  cudaFree(gimage);

  lodepng_encode32_file(output_filename, new_image, width/2, height/2);

  free(image);
  free(new_image);
  printf("Time elapsed = %g ms\n", timer.Elapsed());
  return 0;
}

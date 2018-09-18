#include "../Common/ImageWriter.cuh"
#include "../Common/Vector3.cuh"

const int ImageWidth = 1024;
const int ImageHeight = 512;
const int BlockSize = 1;

// Kernel function to add the elements of two arrays
__global__
void CalculatePixels(int width, int height, Vector3* pixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int numPixels = width * height;
    for (int k = index; k < numPixels; k += stride)
    {
        int j =  height - 1 - (k / width);
        int i = k % width;
        pixels[k] = Vector3(
            static_cast<float>(i) / static_cast<float>(width),
            static_cast<float>(j) / static_cast<float>(height),
            0.2);
    }
}

int main(int argc, char** argv)
{
  std::string fileName = ImageWriter::GetFileName(argc, argv);

  // Allocate Unified Memory – accessible from CPU or GPU
  int numPixels = ImageWidth*ImageHeight;
  Vector3 *pixels;
  cudaMallocManaged(&pixels, numPixels*sizeof(Vector3));

  // Run kernel on the GPU
  int numBlocks = (numPixels + BlockSize - 1) / BlockSize;
  CalculatePixels<<<numBlocks, BlockSize>>>(ImageWidth, ImageHeight, pixels);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  ImageWriter::WritePPM(fileName, ImageWidth, ImageHeight, pixels);

  // Free memory
  cudaFree(pixels);
  
  return 0;
}
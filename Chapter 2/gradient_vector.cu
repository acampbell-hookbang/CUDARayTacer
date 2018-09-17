#include <fstream>
#include <iostream>
#include <string>
#include "Vector3.cuh"

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

void SaveImage(std::string fileName, int width, int height, Vector3* pixels)
{
  std::ofstream imageFile;
  imageFile.open(fileName.c_str());
  imageFile << "P3" << std::endl  << width << " " << height << std::endl << 255 << std::endl;
  int k = 0;
  for (int j = 0; j < height; ++j)
  {
      for (int i = 0; i < width; ++i)
      {
          int ir = static_cast<int>(255.99 * pixels[k].R());
          int ig = static_cast<int>(255.99 * pixels[k].G());
          int ib = static_cast<int>(255.99 * pixels[k].B());
          imageFile << ir << " " << ig << " " << ib << std::endl;
          k++;
      }
  }
  imageFile.close();
}

int main(int argc, char** argv)
{
  // Get image file name
  std::string fileName = "gradient_vector.ppm";
  if (argc > 1)
  {
      fileName = argv[1];
  }

  // Allocate Unified Memory – accessible from CPU or GPU
  int numPixels = ImageWidth*ImageHeight;
  Vector3 *pixels;
  cudaMallocManaged(&pixels, numPixels*sizeof(Vector3));

  // Run kernel on 1M elements on the GPU
  int numBlocks = (numPixels + BlockSize - 1) / BlockSize;
  CalculatePixels<<<numBlocks, BlockSize>>>(ImageWidth, ImageHeight, pixels);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  SaveImage(fileName, ImageWidth, ImageHeight, pixels);

  // Free memory
  cudaFree(pixels);
  
  return 0;
}
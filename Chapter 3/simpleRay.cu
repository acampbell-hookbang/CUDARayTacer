#include "../Common/ImageWriter.cuh"
#include "../Common/Ray.cuh"
#include "../Common/Vector3.cuh"

const int ImageWidth = 1024;
const int ImageHeight = 512;
const int BlockSize = 1;

__host__ __device__ 
Vector3 CalculateColor(const Ray &r)
{
    Vector3 unitDirection = UnitVector(r.Direction());
    float t = 0.5 * (unitDirection.Y() + 1.0);
    t = min(1.0, max(0.0, t));
    return (1.0 - t) * Vector3::One() + t * Vector3(0.5, 0.7, 1.0);
}

__global__
void CalculateImage(int width, int height, Vector3* pixels)
{
    const Vector3 lowerLeft(-2.0, -1.0, -1.0);
    const Vector3 horizontal(4.0, 0.0, 0.0);
    const Vector3 vertical(0.0, 2.0, 0.0);
    const Vector3 origin = Vector3::Zero();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int numPixels = width * height;
    for (int k = index; k < numPixels; k += stride)
    {
        int j =  height - 1 - (k / width);
        int i = k % width;
        float u = static_cast<float>(i) / static_cast<float>(width);
        float v = static_cast<float>(j) / static_cast<float>(height);
        Ray ray(origin, lowerLeft + u * horizontal + v * vertical);
        pixels[k] = CalculateColor(ray);
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
  CalculateImage<<<numBlocks, BlockSize>>>(ImageWidth, ImageHeight, pixels);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  ImageWriter::WritePPM(fileName, ImageWidth, ImageHeight, pixels);

  // Free memory
  cudaFree(pixels);
  
  return 0;
}
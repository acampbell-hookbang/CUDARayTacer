#include "../Common/ImageWriter.cuh"
#include "../Common/Ray.cuh"
#include "../Common/Vector3.cuh"

const int ImageWidth = 1024;
const int ImageHeight = 512;
const int BlockSize = 1;

__host__ __device__ 
float HitSphere(const Vector3 &center, const float radius, const Ray &r)
{
    Vector3 oc = r.Origin() - center;
    float a = Dot(r.Direction(), r.Direction());
    float b = 2.0f * Dot(oc, r.Direction());
    float c = Dot(oc, oc) - radius * radius;
    float discriminant = b*b - 4.0f*a*c;
    if (discriminant < 0.0)
    {
        return -1.0;
    }
    else
    {
        return (-b - sqrt(discriminant))/(2.0f*a);
    }
}

__host__ __device__ 
Vector3 CalculateColor(const Ray &r)
{
    const Vector3 sphereCenter(0.0f, 0.0f, -1.0f);
    const float sphereRadius = 0.5f;
    float t = HitSphere(sphereCenter, sphereRadius, r);
    if (t > 0.0)
    {
        Vector3 normal = UnitVector(r.PointAtParameter(t) - sphereCenter);
        normal.MakeUnitVector();
        return 0.5 * (normal + Vector3::One());
    }
    Vector3 unitDirection = UnitVector(r.Direction());
    t = 0.5f * (unitDirection.Y() + 1.0f);
    t = min(1.0f, max(0.0, t));
    return (1.0f - t) * Vector3::One() + t * Vector3(0.5f, 0.7f, 1.0f);
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
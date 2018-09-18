#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include "../Common/Vector3.cuh"

class ImageWriter
{
    public:
        static std::string GetFileName(int argc, char** argv);
        static void ImageWriter::WritePPM(std::string fileName, int width, int height, Vector3* pixels);
};

std::string ImageWriter::GetFileName(int argc, char** argv)
{
  const std::string executableExtension = ".exe";
  const std::string imageExtension = ".ppm";
  std::string fileName;
  if (argc > 1)
  {
      fileName = std::string(argv[1]);
      int indexExt = fileName.find(imageExtension.c_str(), 0);
      if (indexExt < 0)
      {
        fileName = fileName + imageExtension;
      }
  }
  else
  {
      fileName = std::string(argv[0]);
      int indexExe = fileName.find(executableExtension.c_str(), 0);
      if (indexExe >= 0)
      {
        fileName.replace(indexExe, executableExtension.length(), imageExtension.c_str());
      }
      else
      {
        fileName = std::string(argv[0]) + imageExtension;
      }
  }
  return fileName;
}

void ImageWriter::WritePPM(std::string fileName, int width, int height, Vector3* pixels)
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
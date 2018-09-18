#pragma once
#include "Vector3.cuh"

class Ray
{
    public:
    Vector3 a;
    Vector3 b;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vector3 &org, const Vector3 &dir)
    {
        a = org;
        b = dir;
    }
    __host__ __device__ inline Vector3 Origin() const { return a; }
    __host__ __device__ inline Vector3 Direction() const { return b; }
    __host__ __device__ inline Vector3 PointAtParameter(const float t) const 
    { 
        return a + b * t; 
    }
};
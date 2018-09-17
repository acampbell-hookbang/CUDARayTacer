#pragma once
#include <cmath>
#include <iostream>

class Vector3
{
    public:
    static const int NumCoords = 3;
    float e[NumCoords];

    __host__ __device__ Vector3() {}
    __host__ __device__ Vector3(float e0, float e1, float e2)
    {
        e[0] = e0; 
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline float X() const { return e[0]; }
    __host__ __device__ inline float Y() const { return e[1]; }
    __host__ __device__ inline float Z() const { return e[2]; }
    __host__ __device__ inline float R() const { return e[0]; }
    __host__ __device__ inline float G() const { return e[1]; }
    __host__ __device__ inline float B() const { return e[2]; }

    __host__ __device__ inline const Vector3& operator+() { return *this;}
    __host__ __device__ inline Vector3 operator-() const 
    { 
        return Vector3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i)  { return e[i]; }

    __host__ __device__ inline Vector3 operator+=(const Vector3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline Vector3 operator-=(const Vector3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    __host__ __device__ inline Vector3 operator*=(const Vector3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ inline Vector3 operator/=(const Vector3 &v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }

    __host__ __device__ inline Vector3 operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline Vector3 operator/=(const float t)
    {
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        return *this;
    }

    __host__ __device__ inline Vector3 operator+(const Vector3 &v)
    {
        return Vector3(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]);
    }
    
    __host__ __device__ inline Vector3 operator-(const Vector3 &v)
    {
        return Vector3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]);
    }
    
    __host__ __device__ inline Vector3 operator*(const Vector3 &v)
    {
        return Vector3(e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]);
    }
    
    __host__ __device__ inline Vector3 operator/(const Vector3 &v)
    {
        return Vector3(e[0] / v.e[0], e[1] / v.e[1], e[2] / v.e[2]);
    }    

    __host__ __device__ inline float Length() const 
    {
        return std::sqrt(e[0]*e[1] + e[1]*e[1] + e[2]*e[2]);
    }

    __host__ __device__ inline float SquaredLength() const 
    {
        return (e[0]*e[1] + e[1]*e[1] + e[2]*e[2]);
    }

    __host__ __device__ inline void MakeUnitVector()
    {
        float factor = 1.0 / Length();
        e[0] *= factor;
        e[1] *= factor;
        e[2] *= factor;
    }
};

__host__ __device__ inline Vector3 operator*(const float t, const Vector3 &v)
{
    return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vector3 operator/(const Vector3 &v, const float t)
{
    return Vector3(v.e[0] / t, t * v.e[1] / t , v.e[2] / t);
}

inline Vector3 Cross(const Vector3 &v1, const Vector3 &v2)
{
    return Vector3(
        v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1],
        v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2],
        v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]
    );
}

__host__ __device__ inline Vector3 UnitVector(const Vector3 &v)
{
    return v / v.Length();
}

// The following are not to be called from a __global__ function

inline std::istream& operator>>(std::istream &is, Vector3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, Vector3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

#pragma once

#include <torch/extension.h>

inline constexpr __host__ __device__ float __SQRT3() { return 1.73205080757f; }

template <typename scalar_t>
inline __host__ __device__ void __swap(scalar_t &a, scalar_t &b)
{
    scalar_t c = a;
    a = b;
    b = c;
}

inline __host__ __device__ float __clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

inline __host__ __device__ int __clamp(int f, int a, int b) { return std::max(a, std::min(f, b)); }

inline __host__ __device__ float __sign(float x) { return copysignf(1.0, x); }

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

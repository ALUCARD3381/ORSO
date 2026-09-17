#pragma once

#include <cstddef>

namespace orso::kernels {

void add_f32(const float* a, const float* b, float* out, std::size_t n);
void sub_f32(const float* a, const float* b, float* out, std::size_t n);
void mul_f32(const float* a, const float* b, float* out, std::size_t n);
float dot_f32(const float* a, const float* b, std::size_t n);
void matmul_f32(const float* a, const float* b, float* out, std::size_t M, std::size_t K, std::size_t N);

} // namespace orso::kernels

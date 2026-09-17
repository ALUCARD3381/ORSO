#pragma once

#include <cstddef>

namespace orso::neon {

bool available() noexcept;

void add(const float* a, const float* b, float* out, std::size_t n) noexcept;
void sub(const float* a, const float* b, float* out, std::size_t n) noexcept;
void mul(const float* a, const float* b, float* out, std::size_t n) noexcept;
float dot(const float* a, const float* b, std::size_t n) noexcept;
void matmul_2d(const float* a, const float* b, float* out,
               std::size_t m, std::size_t k, std::size_t n) noexcept;

} // namespace orso::neon

#include "neon_kernels.hpp"

#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif

namespace orso::neon {

bool available() noexcept {
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    return true;
#else
    return false;
#endif
}

void add(const float* a, const float* b, float* out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

void sub(const float* a, const float* b, float* out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

void mul(const float* a, const float* b, float* out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

float dot(const float* a, const float* b, std::size_t n) noexcept {
    std::size_t i = 0;
    float sum = 0.0f;
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float lanes[4];
    vst1q_f32(lanes, acc);
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

void matmul_2d(const float* a, const float* b, float* out,
               std::size_t m, std::size_t k, std::size_t n) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        float* out_row = out + i * n;
        const float* a_row = a + i * k;
        std::size_t j = 0;
#if defined(ORSO_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        for (; j + 4 <= n; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (std::size_t p = 0; p < k; ++p) {
                const float32x4_t bv = vld1q_f32(b + p * n + j);
                acc = vmlaq_n_f32(acc, bv, a_row[p]);
            }
            vst1q_f32(out_row + j, acc);
        }
#endif
        for (; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                sum += a_row[p] * b[p * n + j];
            }
            out_row[j] = sum;
        }
    }
}

} // namespace orso::neon

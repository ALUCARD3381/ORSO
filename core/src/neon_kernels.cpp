#include "orso/neon_kernels.hpp"

#ifdef ORSO_USE_NEON
#include <arm_neon.h>
#endif

namespace orso::kernels {

void add_f32(const float* a, const float* b, float* out, std::size_t n) {
#ifdef ORSO_USE_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
#else
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

void sub_f32(const float* a, const float* b, float* out, std::size_t n) {
#ifdef ORSO_USE_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] - b[i];
#else
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
#endif
}

void mul_f32(const float* a, const float* b, float* out, std::size_t n) {
#ifdef ORSO_USE_NEON
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
#else
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

float dot_f32(const float* a, const float* b, std::size_t n) {
#ifdef ORSO_USE_NEON
    std::size_t i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float lanes[4];
    vst1q_f32(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void matmul_f32(const float* a, const float* b, float* out, std::size_t M, std::size_t K, std::size_t N) {
#ifdef ORSO_USE_NEON
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 4 <= N; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (std::size_t k = 0; k < K; ++k) {
                const float32x4_t bv = vld1q_f32(b + k * N + j);
                const float32x4_t av = vmulq_n_f32(bv, a[i * K + k]);
                acc = vaddq_f32(acc, av);
            }
            vst1q_f32(out + i * N + j, acc);
        }
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) sum += a[i * K + k] * b[k * N + j];
            out[i * N + j] = sum;
        }
    }
#else
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) sum += a[i * K + k] * b[k * N + j];
            out[i * N + j] = sum;
        }
    }
#endif
}

} // namespace orso::kernels

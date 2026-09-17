#pragma once

#include "orso/tensor.hpp"

namespace orso {

Tensor causal_mask(std::size_t batch, std::size_t heads, std::size_t seq_len);
Tensor multi_head_attention(const Tensor& x,
                            const Tensor& wq,
                            const Tensor& wk,
                            const Tensor& wv,
                            const Tensor& wo,
                            std::size_t num_heads,
                            bool causal = true,
                            float rope_theta = 10000.0f);
Tensor swiglu(const Tensor& x,
              const Tensor& gate_weight,
              const Tensor& up_weight,
              const Tensor& down_weight);

} // namespace orso

#pragma once

#include "orso/tensor.hpp"
#include <cstddef>
#include <vector>

namespace orso {

class AdamW {
public:
    explicit AdamW(const std::vector<Tensor>& parameters,
                   float lr = 3.0e-4f,
                   float beta1 = 0.9f,
                   float beta2 = 0.999f,
                   float eps = 1.0e-8f,
                   float weight_decay = 0.01f,
                   float max_grad_norm = 0.0f);

    void zero_grad();
    float step();

    float lr() const { return lr_; }
    void set_lr(float value);
    std::size_t step_count() const { return step_count_; }

private:
    std::vector<Tensor> parameters_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    float max_grad_norm_;
    std::size_t step_count_{0};
};

} // namespace orso

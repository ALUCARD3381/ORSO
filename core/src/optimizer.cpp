#include "orso/optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace orso {

AdamW::AdamW(const std::vector<Tensor>& parameters,
             float lr,
             float beta1,
             float beta2,
             float eps,
             float weight_decay,
             float max_grad_norm)
    : parameters_(parameters),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      max_grad_norm_(max_grad_norm) {
    if (parameters_.empty()) throw std::invalid_argument("AdamW requires at least one parameter");
    if (!(lr_ > 0.0f)) throw std::invalid_argument("AdamW lr must be > 0");
    if (!(beta1_ >= 0.0f && beta1_ < 1.0f)) throw std::invalid_argument("AdamW beta1 must be in [0,1)");
    if (!(beta2_ >= 0.0f && beta2_ < 1.0f)) throw std::invalid_argument("AdamW beta2 must be in [0,1)");
    if (!(eps_ > 0.0f)) throw std::invalid_argument("AdamW eps must be > 0");
    if (weight_decay_ < 0.0f) throw std::invalid_argument("AdamW weight_decay must be >= 0");
    if (max_grad_norm_ < 0.0f) throw std::invalid_argument("AdamW max_grad_norm must be >= 0");

    m_.reserve(parameters_.size());
    v_.reserve(parameters_.size());
    for (const Tensor& p : parameters_) {
        if (!p.requires_grad()) throw std::invalid_argument("all AdamW parameters must require_grad");
        m_.emplace_back(p.size(), 0.0f);
        v_.emplace_back(p.size(), 0.0f);
    }
}

void AdamW::set_lr(float value) {
    if (!(value > 0.0f)) throw std::invalid_argument("AdamW lr must be > 0");
    lr_ = value;
}

void AdamW::zero_grad() {
    for (Tensor& p : parameters_) p.zero_grad();
}

float AdamW::step() {
    float sum_sq = 0.0f;
    for (const Tensor& p : parameters_) {
        for (float g : p.grad()) sum_sq += g * g;
    }
    const float grad_norm = std::sqrt(sum_sq);
    float clip_scale = 1.0f;
    if (max_grad_norm_ > 0.0f && grad_norm > max_grad_norm_ && grad_norm > 0.0f) {
        clip_scale = max_grad_norm_ / grad_norm;
    }

    ++step_count_;
    const float bias1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    const float bias2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (std::size_t pidx = 0; pidx < parameters_.size(); ++pidx) {
        Tensor& p = parameters_[pidx];
        auto& m = m_[pidx];
        auto& v = v_[pidx];
        auto& data = p.mutable_data();
        const auto& grad = p.grad();

        for (std::size_t i = 0; i < data.size(); ++i) {
            const float g = grad[i] * clip_scale;
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g;
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g * g;
            const float mhat = m[i] / bias1;
            const float vhat = v[i] / bias2;
            const float adam_step = mhat / (std::sqrt(vhat) + eps_);
            const float decay = weight_decay_ * data[i];
            data[i] -= lr_ * (adam_step + decay);
        }
    }
    return grad_norm;
}

std::vector<std::vector<float>> AdamW::first_moment() const {
    return m_;
}

std::vector<std::vector<float>> AdamW::second_moment() const {
    return v_;
}

void AdamW::load_state(const std::vector<std::vector<float>>& first,
                       const std::vector<std::vector<float>>& second,
                       std::size_t step_count) {
    if (first.size() != parameters_.size() || second.size() != parameters_.size())
        throw std::invalid_argument("AdamW state parameter count mismatch");
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        if (first[i].size() != parameters_[i].size() || second[i].size() != parameters_[i].size())
            throw std::invalid_argument("AdamW state parameter size mismatch");
        for (float value : first[i]) {
            if (!std::isfinite(value)) throw std::invalid_argument("AdamW first moment contains non-finite value");
        }
        for (float value : second[i]) {
            if (!std::isfinite(value)) throw std::invalid_argument("AdamW second moment contains non-finite value");
        }
    }
    m_ = first;
    v_ = second;
    step_count_ = step_count;
}

} // namespace orso

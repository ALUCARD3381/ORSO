#include "orso/model.hpp"
#include "orso/transformer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace orso {
namespace {
void validate_config(const ModelConfig& c) {
    if (c.vocab_size < 2) throw std::invalid_argument("vocab_size must be >= 2");
    if (c.d_model == 0) throw std::invalid_argument("d_model must be > 0");
    if (c.num_heads == 0 || c.d_model % c.num_heads != 0)
        throw std::invalid_argument("d_model must be divisible by num_heads");
    if (c.hidden_dim == 0) throw std::invalid_argument("hidden_dim must be > 0");
    if (c.num_layers == 0) throw std::invalid_argument("num_layers must be > 0");
    if (c.context_length == 0) throw std::invalid_argument("context_length must be > 0");
}
Tensor init_matrix(const Shape& shape, std::uint64_t& seed, float stddev) {
    return Tensor::random_normal(shape, 0.0f, stddev, seed++, true);
}
Tensor init_norm(std::size_t d) {
    return Tensor::ones({d}, true);
}

Tensor detach_tensor(const Tensor& x) {
    return Tensor(x.shape(), x.data(), false);
}

Tensor rope_at_position(const Tensor& x, std::size_t position, float theta) {
    if (x.ndim() != 4 || x.shape()[2] != 1 || (x.shape()[3] % 2) != 0)
        throw std::invalid_argument("cached RoPE expects [B,H,1,Dh] with even Dh");
    const std::size_t B = x.shape()[0];
    const std::size_t H = x.shape()[1];
    const std::size_t D = x.shape()[3];
    std::vector<float> out(x.size());
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t base = (b * H + h) * D;
            for (std::size_t j = 0; j < D; j += 2) {
                const float inv_freq = std::pow(theta, -static_cast<float>(j) / static_cast<float>(D));
                const float angle = static_cast<float>(position) * inv_freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float xe = x.data()[base + j];
                const float xo = x.data()[base + j + 1];
                out[base + j] = xe * c - xo * s;
                out[base + j + 1] = xe * s + xo * c;
            }
        }
    }
    return Tensor(x.shape(), out, false);
}

Tensor append_time(const Tensor& existing, const Tensor& token) {
    if (token.ndim() != 4 || token.shape()[0] != 1 || token.shape()[2] != 1)
        throw std::invalid_argument("KV append expects token shape [1,H,1,Dh]");
    if (existing.shape().empty()) return detach_tensor(token);
    if (existing.ndim() != 4 || existing.shape()[0] != 1 || existing.shape()[1] != token.shape()[1] ||
        existing.shape()[3] != token.shape()[3])
        throw std::invalid_argument("KV cache shape mismatch");
    const std::size_t H = token.shape()[1];
    const std::size_t T = existing.shape()[2];
    const std::size_t D = token.shape()[3];
    std::vector<float> out(existing.size() + token.size());
    for (std::size_t h = 0; h < H; ++h) {
        std::copy_n(existing.data().data() + h * T * D, T * D, out.data() + h * (T + 1) * D);
        std::copy_n(token.data().data() + h * D, D, out.data() + h * (T + 1) * D + T * D);
    }
    return Tensor({1, H, T + 1, D}, out, false);
}

} // namespace

TransformerModel::TransformerModel(const ModelConfig& config) : config_(config) {
    validate_config(config_);
    std::uint64_t seed = config_.seed;
    embedding_weight_ = init_matrix({config_.vocab_size, config_.d_model}, seed, 0.02f);
    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) {
        Block block;
        block.norm1 = init_norm(config_.d_model);
        block.wq = init_matrix({config_.d_model, config_.d_model}, seed, 0.02f);
        block.wk = init_matrix({config_.d_model, config_.d_model}, seed, 0.02f);
        block.wv = init_matrix({config_.d_model, config_.d_model}, seed, 0.02f);
        block.wo = init_matrix({config_.d_model, config_.d_model}, seed, 0.02f);
        block.norm2 = init_norm(config_.d_model);
        block.wgate = init_matrix({config_.d_model, config_.hidden_dim}, seed, 0.02f);
        block.wup = init_matrix({config_.d_model, config_.hidden_dim}, seed, 0.02f);
        block.wdown = init_matrix({config_.hidden_dim, config_.d_model}, seed, 0.02f);
        blocks_.push_back(std::move(block));
    }
    final_norm_ = init_norm(config_.d_model);
    lm_head_ = init_matrix({config_.d_model, config_.vocab_size}, seed, 0.02f);
}

Tensor TransformerModel::forward(const std::vector<std::vector<int>>& token_ids) const {
    if (token_ids.empty()) throw std::invalid_argument("token_ids cannot be empty");
    const std::size_t batch = token_ids.size();
    const std::size_t seq = token_ids[0].size();
    if (seq == 0) throw std::invalid_argument("token sequence cannot be empty");
    if (seq > config_.context_length)
        throw std::invalid_argument("sequence length exceeds context_length");
    for (const auto& row : token_ids) {
        if (row.size() != seq) throw std::invalid_argument("token_ids must be rectangular");
    }
    Tensor h = embedding_lookup(embedding_weight_, token_ids);
    for (const auto& block : blocks_) {
        Tensor n1 = rmsnorm(h, block.norm1);
        Tensor attn = multi_head_attention(
            n1, block.wq, block.wk, block.wv, block.wo, config_.num_heads, true);
        h = add(h, attn);
        Tensor n2 = rmsnorm(h, block.norm2);
        Tensor ffn = swiglu(n2, block.wgate, block.wup, block.wdown);
        h = add(h, ffn);
    }
    h = rmsnorm(h, final_norm_);
    return matmul(h, lm_head_);
}

void TransformerModel::reset_kv_cache() {
    kv_cache_.clear();
    kv_tokens_.clear();
}

std::size_t TransformerModel::kv_cache_length() const {
    return kv_tokens_.size();
}

Tensor TransformerModel::forward_cached(const std::vector<int>& token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("token_ids cannot be empty");
    for (int token_id : token_ids) {
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= config_.vocab_size)
            throw std::out_of_range("token id outside vocabulary");
    }
    Tensor last;
    for (int token_id : token_ids) last = forward_cached_one(token_id);
    return last;
}

Tensor TransformerModel::forward_cached_one(int token_id) {
    if (kv_cache_.size() != blocks_.size()) reset_kv_cache();

    // Once the window is full, rebuild from the last context_length - 1 tokens
    // plus the incoming token. This matches the non-cached model, whose RoPE
    // positions restart at zero for the active sliding window.
    if (kv_tokens_.size() >= config_.context_length) {
        std::vector<int> rebuild;
        const std::size_t keep = config_.context_length - 1;
        rebuild.insert(rebuild.end(), kv_tokens_.end() - static_cast<std::ptrdiff_t>(keep), kv_tokens_.end());
        rebuild.push_back(token_id);
        reset_kv_cache();
        Tensor last;
        for (int t : rebuild) last = forward_cached_one(t);
        return last;
    }

    const std::size_t position = kv_tokens_.size();
    const std::size_t H = config_.num_heads;
    const std::size_t D = config_.d_model;
    const std::size_t Dh = D / H;

    Tensor h = embedding_lookup(embedding_weight_, {{token_id}});

    if (kv_cache_.empty()) kv_cache_.resize(blocks_.size());
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& block = blocks_[layer];
        Tensor n1 = rmsnorm(h, block.norm1);
        Tensor q = matmul(n1, block.wq).reshape({1, 1, H, Dh}).transpose({0, 2, 1, 3});
        Tensor k = matmul(n1, block.wk).reshape({1, 1, H, Dh}).transpose({0, 2, 1, 3});
        Tensor v = matmul(n1, block.wv).reshape({1, 1, H, Dh}).transpose({0, 2, 1, 3});
        q = rope_at_position(q, position, 10000.0f);
        k = rope_at_position(k, position, 10000.0f);

        kv_cache_[layer].key = append_time(kv_cache_[layer].key, k);
        kv_cache_[layer].value = append_time(kv_cache_[layer].value, v);

        Tensor kt = kv_cache_[layer].key.transpose({0, 1, 3, 2});
        Tensor scores = div(matmul(detach_tensor(q), kt), std::sqrt(static_cast<float>(Dh)));
        Tensor probs = softmax(scores, -1);
        Tensor ctx = matmul(probs, kv_cache_[layer].value);
        ctx = ctx.transpose({0, 2, 1, 3}).reshape({1, 1, D});
        Tensor attn = matmul(ctx, block.wo);
        h = add(h, attn);

        Tensor n2 = rmsnorm(h, block.norm2);
        Tensor ffn = swiglu(n2, block.wgate, block.wup, block.wdown);
        h = add(h, ffn);
    }

    kv_tokens_.push_back(token_id);
    h = rmsnorm(h, final_norm_);
    return matmul(h, lm_head_);
}

std::vector<Tensor> TransformerModel::parameters() const {
    std::vector<Tensor> out;
    out.reserve(2 + blocks_.size() * 9);
    out.push_back(embedding_weight_);
    for (const auto& block : blocks_) {
        out.push_back(block.norm1);
        out.push_back(block.wq);
        out.push_back(block.wk);
        out.push_back(block.wv);
        out.push_back(block.wo);
        out.push_back(block.norm2);
        out.push_back(block.wgate);
        out.push_back(block.wup);
        out.push_back(block.wdown);
    }
    out.push_back(final_norm_);
    out.push_back(lm_head_);
    return out;
}

std::size_t TransformerModel::parameter_count() const {
    std::size_t total = 0;
    for (const Tensor& p : parameters()) total += p.size();
    return total;
}

std::vector<std::vector<float>> TransformerModel::parameter_data() const {
    std::vector<std::vector<float>> out;
    const auto params = parameters();
    out.reserve(params.size());
    for (const Tensor& p : params) out.push_back(p.data());
    return out;
}

void TransformerModel::load_parameter_data(const std::vector<std::vector<float>>& values) {
    auto params = parameters();
    if (values.size() != params.size())
        throw std::invalid_argument("parameter count mismatch while loading model state");
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (values[i].size() != params[i].size())
            throw std::invalid_argument("parameter size mismatch while loading model state");
        for (float value : values[i]) {
            if (!std::isfinite(value)) throw std::invalid_argument("model state contains non-finite value");
        }
        auto& dst = params[i].mutable_data();
        std::copy(values[i].begin(), values[i].end(), dst.begin());
        params[i].zero_grad();
    }
    reset_kv_cache();
}

} // namespace orso

#include "orso/model.hpp"
#include "orso/transformer.hpp"
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

} // namespace orso

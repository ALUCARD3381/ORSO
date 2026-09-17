#pragma once
#include "orso/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace orso {

struct ModelConfig {
    std::size_t vocab_size{768};
    std::size_t d_model{64};
    std::size_t num_heads{8};
    std::size_t hidden_dim{256};
    std::size_t num_layers{6};
    std::size_t context_length{32};
    std::uint64_t seed{1234};
};

class TransformerModel {
public:
    explicit TransformerModel(const ModelConfig& config = {});
    Tensor forward(const std::vector<std::vector<int>>& token_ids) const;

    // Inference-only incremental path. The model keeps one KV cache per layer.
    // The cache is automatically reset/rebuilt when the context window slides.
    Tensor forward_cached(const std::vector<int>& token_ids);
    void reset_kv_cache();
    std::size_t kv_cache_length() const;

    std::vector<Tensor> parameters() const;
    std::vector<std::vector<float>> parameter_data() const;
    void load_parameter_data(const std::vector<std::vector<float>>& values);
    std::size_t parameter_count() const;
    const ModelConfig& config() const { return config_; }
private:
    struct Block {
        Tensor norm1;
        Tensor wq;
        Tensor wk;
        Tensor wv;
        Tensor wo;
        Tensor norm2;
        Tensor wgate;
        Tensor wup;
        Tensor wdown;
    };

    struct KVCacheEntry {
        Tensor key;
        Tensor value;
    };

    Tensor forward_cached_one(int token_id);

    ModelConfig config_;
    Tensor embedding_weight_;
    std::vector<Block> blocks_;
    Tensor final_norm_;
    Tensor lm_head_;

    std::vector<KVCacheEntry> kv_cache_;
    std::vector<int> kv_tokens_;
};

} // namespace orso

#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace orso {

class Tensor {
public:
    Tensor();
    explicit Tensor(const std::vector<std::size_t>& shape);
    Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& data);

    static Tensor zeros(const std::vector<std::size_t>& shape);
    static Tensor ones(const std::vector<std::size_t>& shape);
    static Tensor full(const std::vector<std::size_t>& shape, float value);

    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    const std::vector<float>& data() const noexcept { return data_; }
    std::vector<float>& data() noexcept { return data_; }

    float item() const;
    float get(const std::vector<std::size_t>& indices) const;
    void set(const std::vector<std::size_t>& indices, float value);

    Tensor reshape(const std::vector<std::size_t>& new_shape) const;
    Tensor transpose() const;
    Tensor transpose(const std::vector<std::size_t>& axes) const;

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor div(float scalar) const;
    Tensor neg() const;

    Tensor matmul(const Tensor& other) const;

    void fill(float value);

    std::string repr() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> data_;

    static std::size_t checked_numel(const std::vector<std::size_t>& shape);
    static std::vector<std::size_t> make_contiguous_strides(const std::vector<std::size_t>& shape);
    static void validate_shape(const std::vector<std::size_t>& shape);
    void rebuild_strides();
    std::size_t offset(const std::vector<std::size_t>& indices) const;

    Tensor elementwise_same_shape(const Tensor& other, char op) const;
    static std::size_t flat_batch_count(const std::vector<std::size_t>& batch_shape);
};

} // namespace orso

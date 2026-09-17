#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace orso {

class Tensor;

struct AutogradNode;
struct AutogradMeta;

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

    bool requires_grad() const noexcept;
    void set_requires_grad(bool value);
    bool has_grad() const noexcept;
    Tensor grad() const;
    void zero_grad();
    void backward();
    void backward(const Tensor& grad);

    void fill(float value);

    std::string repr() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> data_;
    std::shared_ptr<AutogradMeta> autograd_;

    static std::size_t checked_numel(const std::vector<std::size_t>& shape);
    static std::vector<std::size_t> make_contiguous_strides(const std::vector<std::size_t>& shape);
    static void validate_shape(const std::vector<std::size_t>& shape);
    void rebuild_strides();
    std::size_t offset(const std::vector<std::size_t>& indices) const;

    Tensor elementwise_same_shape(const Tensor& other, char op) const;
    static std::size_t flat_batch_count(const std::vector<std::size_t>& batch_shape);

    void attach_grad_fn(std::shared_ptr<AutogradNode> node);
    static void accumulate_grad(const std::shared_ptr<AutogradMeta>& meta, const Tensor& contribution);
    static std::vector<Tensor> topo_sort(const Tensor& root);
};

struct AutogradNode {
    std::vector<Tensor> parents;
    std::function<std::vector<Tensor>(const Tensor&)> backward;
};

struct AutogradMeta {
    bool requires_grad{false};
    std::vector<float> grad;
    std::shared_ptr<AutogradNode> grad_fn;
};

} // namespace orso

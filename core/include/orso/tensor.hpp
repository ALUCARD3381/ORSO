#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace orso {

using Shape = std::vector<std::size_t>;

struct TensorImpl;

struct BackwardNode {
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(const std::vector<float>&)> backward;
};

struct TensorImpl {
    Shape shape;
    std::vector<float> data;
    bool requires_grad{false};
    std::vector<float> grad;
    std::shared_ptr<BackwardNode> grad_fn;
};

class Tensor {
public:
    Tensor();
    explicit Tensor(const Shape& shape, float fill = 0.0f, bool requires_grad = false);
    Tensor(const Shape& shape, const std::vector<float>& values, bool requires_grad = false);

    static Tensor zeros(const Shape& shape, bool requires_grad = false);
    static Tensor ones(const Shape& shape, bool requires_grad = false);
    static Tensor full(const Shape& shape, float value, bool requires_grad = false);
    static Tensor random_normal(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                                unsigned long long seed = 0, bool requires_grad = false);

    const Shape& shape() const;
    std::size_t ndim() const;
    std::size_t size() const;
    bool requires_grad() const;
    void set_requires_grad(bool value);

    const std::vector<float>& data() const;
    std::vector<float>& mutable_data();
    float item() const;

    float get(const std::vector<std::size_t>& index) const;
    void set(const std::vector<std::size_t>& index, float value);

    const std::vector<float>& grad() const;
    void zero_grad();
    void backward(const Tensor* grad_output = nullptr);

    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(const std::vector<std::size_t>& permutation) const;

    Tensor sum() const;
    Tensor mean() const;

    std::string repr() const;

    std::shared_ptr<TensorImpl> impl() const { return impl_; }
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

private:
    std::shared_ptr<TensorImpl> impl_;

    friend Tensor add(const Tensor&, const Tensor&);
    friend Tensor sub(const Tensor&, const Tensor&);
    friend Tensor mul(const Tensor&, const Tensor&);
    friend Tensor div(const Tensor&, const Tensor&);
    friend Tensor neg(const Tensor&);
    friend Tensor exp(const Tensor&);
    friend Tensor log(const Tensor&);
    friend Tensor sqrt(const Tensor&);
    friend Tensor silu(const Tensor&);
    friend Tensor matmul(const Tensor&, const Tensor&);
    friend Tensor softmax(const Tensor&, int);
    friend Tensor rmsnorm(const Tensor&, const Tensor&, float);
    friend Tensor rope(const Tensor&, float);
    friend Tensor embedding_lookup(const Tensor&, const std::vector<std::vector<int>>&);
};

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, float scalar);
Tensor sub(const Tensor& a, float scalar);
Tensor mul(const Tensor& a, float scalar);
Tensor div(const Tensor& a, float scalar);
Tensor neg(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor silu(const Tensor& a);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& x, int axis = -1);
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);
Tensor rope(const Tensor& x, float theta = 10000.0f);
Tensor embedding_lookup(const Tensor& weight, const std::vector<std::vector<int>>& token_ids);

void ensure_same_shape(const Tensor& a, const Tensor& b, const char* op);
void accumulate_grad(const std::shared_ptr<TensorImpl>& target, const std::vector<float>& grad);

} // namespace orso

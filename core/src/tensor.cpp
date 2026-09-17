#include "tensor.hpp"
#include "neon_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace orso {

namespace {

thread_local bool g_autograd_enabled = true;

struct NoGradGuard {
    bool previous;
    NoGradGuard() noexcept : previous(g_autograd_enabled) { g_autograd_enabled = false; }
    ~NoGradGuard() { g_autograd_enabled = previous; }
};

Tensor transpose_last2(const Tensor& t) {
    if (t.ndim() < 2) {
        throw std::invalid_argument("transpose_last2 requires rank >= 2");
    }
    std::vector<std::size_t> axes(t.ndim());
    std::iota(axes.begin(), axes.end(), 0);
    std::swap(axes[t.ndim() - 1], axes[t.ndim() - 2]);
    return t.transpose(axes);
}

void ensure_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(op) + " requires equal tensor shapes");
    }
}

} // namespace

Tensor::Tensor() : shape_{}, strides_{}, data_{0.0f}, autograd_(std::make_shared<AutogradMeta>()) {}

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape), data_(checked_numel(shape), 0.0f), autograd_(std::make_shared<AutogradMeta>()) {
    validate_shape(shape_);
    rebuild_strides();
}

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data), autograd_(std::make_shared<AutogradMeta>()) {
    validate_shape(shape_);
    const auto expected = checked_numel(shape_);
    if (expected != data_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape");
    }
    rebuild_strides();
}

Tensor Tensor::zeros(const std::vector<std::size_t>& shape) { return Tensor(shape); }
Tensor Tensor::ones(const std::vector<std::size_t>& shape) { return full(shape, 1.0f); }

Tensor Tensor::full(const std::vector<std::size_t>& shape, float value) {
    Tensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), value);
    return t;
}

void Tensor::validate_shape(const std::vector<std::size_t>& shape) {
    (void)shape;
}

std::size_t Tensor::checked_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) return 1;
    constexpr auto max_size = std::numeric_limits<std::size_t>::max();
    std::size_t total = 1;
    for (const auto dim : shape) {
        if (dim != 0 && total > max_size / dim) {
            throw std::overflow_error("Tensor size overflows size_t");
        }
        total *= dim;
    }
    return total;
}

std::vector<std::size_t> Tensor::make_contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    if (shape.empty()) return strides;
    for (std::size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

void Tensor::rebuild_strides() { strides_ = make_contiguous_strides(shape_); }

std::size_t Tensor::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index rank does not match tensor rank");
    }
    std::size_t result = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) throw std::out_of_range("Tensor index out of range");
        result += indices[i] * strides_[i];
    }
    return result;
}

float Tensor::item() const {
    if (data_.size() != 1) throw std::invalid_argument("item() requires a tensor with exactly one element");
    return data_[0];
}

float Tensor::get(const std::vector<std::size_t>& indices) const { return data_[offset(indices)]; }
void Tensor::set(const std::vector<std::size_t>& indices, float value) { data_[offset(indices)] = value; }

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
    const auto expected = checked_numel(new_shape);
    if (expected != data_.size()) throw std::invalid_argument("reshape cannot change the number of elements");
    Tensor out(new_shape, data_);
    if (requires_grad()) {
        auto node = std::make_shared<AutogradNode>();
        node->parents = {*this};
        const auto old_shape = shape_;
        node->backward = [old_shape](const Tensor& grad) {
            return std::vector<Tensor>{grad.reshape(old_shape)};
        };
        out.attach_grad_fn(node);
    }
    return out;
}

Tensor Tensor::transpose() const {
    std::vector<std::size_t> axes(shape_.size());
    std::iota(axes.begin(), axes.end(), 0);
    std::reverse(axes.begin(), axes.end());
    return transpose(axes);
}

Tensor Tensor::transpose(const std::vector<std::size_t>& axes) const {
    if (axes.size() != shape_.size()) {
        throw std::invalid_argument("transpose axes rank does not match tensor rank");
    }
    std::vector<bool> seen(shape_.size(), false);
    for (const auto axis : axes) {
        if (axis >= shape_.size() || seen[axis]) {
            throw std::invalid_argument("transpose axes must be a permutation of tensor dimensions");
        }
        seen[axis] = true;
    }

    std::vector<std::size_t> new_shape(shape_.size());
    for (std::size_t i = 0; i < axes.size(); ++i) new_shape[i] = shape_[axes[i]];

    Tensor out(new_shape);
    if (!data_.empty()) {
        std::vector<std::size_t> new_indices(new_shape.size(), 0);
        std::vector<std::size_t> old_indices(shape_.size(), 0);
        for (std::size_t linear = 0; linear < out.size(); ++linear) {
            std::size_t remainder = linear;
            for (std::size_t i = 0; i < new_shape.size(); ++i) {
                const auto stride = out.strides_[i];
                new_indices[i] = stride == 0 ? 0 : remainder / stride;
                remainder = stride == 0 ? 0 : remainder % stride;
                old_indices[axes[i]] = new_indices[i];
            }
            out.data_[linear] = data_[offset(old_indices)];
        }
    }

    if (requires_grad()) {
        auto node = std::make_shared<AutogradNode>();
        node->parents = {*this};
        std::vector<std::size_t> inverse(axes.size());
        for (std::size_t i = 0; i < axes.size(); ++i) inverse[axes[i]] = i;
        node->backward = [inverse](const Tensor& grad) {
            return std::vector<Tensor>{grad.transpose(inverse)};
        };
        out.attach_grad_fn(node);
    }
    return out;
}

Tensor Tensor::elementwise_same_shape(const Tensor& other, char op) const {
    ensure_same_shape(*this, other, "Elementwise operation");
    Tensor out(shape_);
    switch (op) {
        case '+': neon::add(data_.data(), other.data_.data(), out.data_.data(), data_.size()); break;
        case '-': neon::sub(data_.data(), other.data_.data(), out.data_.data(), data_.size()); break;
        case '*': neon::mul(data_.data(), other.data_.data(), out.data_.data(), data_.size()); break;
        case '/':
            for (std::size_t i = 0; i < data_.size(); ++i) {
                if (other.data_[i] == 0.0f) throw std::domain_error("Tensor division by zero");
                out.data_[i] = data_[i] / other.data_[i];
            }
            break;
        default: throw std::invalid_argument("Unknown elementwise operation");
    }

    if (requires_grad() || other.requires_grad()) {
        auto node = std::make_shared<AutogradNode>();
        node->parents = {*this, other};
        if (op == '+') {
            node->backward = [](const Tensor& grad) { return std::vector<Tensor>{grad, grad}; };
        } else if (op == '-') {
            node->backward = [](const Tensor& grad) { return std::vector<Tensor>{grad, grad.neg()}; };
        } else if (op == '*') {
            const Tensor a = *this;
            const Tensor b = other;
            node->backward = [a, b](const Tensor& grad) {
                return std::vector<Tensor>{grad.mul(b), grad.mul(a)};
            };
        } else {
            const Tensor a = *this;
            const Tensor b = other;
            node->backward = [a, b](const Tensor& grad) {
                const Tensor db = b.mul(b);
                const Tensor ga = grad.div(b);
                const Tensor gb = grad.mul(a).div(db).neg();
                return std::vector<Tensor>{ga, gb};
            };
        }
        out.attach_grad_fn(node);
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const { return elementwise_same_shape(other, '+'); }
Tensor Tensor::sub(const Tensor& other) const { return elementwise_same_shape(other, '-'); }
Tensor Tensor::mul(const Tensor& other) const { return elementwise_same_shape(other, '*'); }
Tensor Tensor::div(const Tensor& other) const { return elementwise_same_shape(other, '/'); }

Tensor Tensor::mul(float scalar) const {
    Tensor out(shape_);
    for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] * scalar;
    if (requires_grad()) {
        auto node = std::make_shared<AutogradNode>();
        node->parents = {*this};
        node->backward = [scalar](const Tensor& grad) { return std::vector<Tensor>{grad.mul(scalar)}; };
        out.attach_grad_fn(node);
    }
    return out;
}

Tensor Tensor::div(float scalar) const {
    if (scalar == 0.0f) throw std::domain_error("Tensor division by zero");
    Tensor out = mul(1.0f / scalar);
    return out;
}

Tensor Tensor::neg() const { return mul(-1.0f); }

std::size_t Tensor::flat_batch_count(const std::vector<std::size_t>& batch_shape) {
    return checked_numel(batch_shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    const auto ra = ndim();
    const auto rb = other.ndim();
    if (ra == 0 || rb == 0) throw std::invalid_argument("matmul requires tensors with rank >= 1");

    Tensor out;

    if (ra == 1 && rb == 1) {
        if (shape_[0] != other.shape_[0]) throw std::invalid_argument("matmul dimension mismatch");
        out = Tensor(std::vector<std::size_t>{});
        out.data_[0] = neon::dot(data_.data(), other.data_.data(), shape_[0]);
    } else if (ra == 2 && rb == 1) {
        const auto m = shape_[0];
        const auto k = shape_[1];
        if (k != other.shape_[0]) throw std::invalid_argument("matmul dimension mismatch");
        out = Tensor({m});
        for (std::size_t i = 0; i < m; ++i) out.data_[i] = neon::dot(data_.data() + i * k, other.data_.data(), k);
    } else if (ra == 1 && rb == 2) {
        const auto k = shape_[0];
        const auto n = other.shape_[1];
        if (k != other.shape_[0]) throw std::invalid_argument("matmul dimension mismatch");
        out = Tensor({n});
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < k; ++i) sum += data_[i] * other.data_[i * n + j];
            out.data_[j] = sum;
        }
    } else if (ra >= 2 && rb >= 2) {
        const auto a_m = shape_[ra - 2];
        const auto a_k = shape_[ra - 1];
        const auto b_k = other.shape_[rb - 2];
        const auto b_n = other.shape_[rb - 1];
        if (a_k != b_k) throw std::invalid_argument("matmul inner dimensions mismatch");
        const std::vector<std::size_t> a_batch(shape_.begin(), shape_.end() - 2);
        const std::vector<std::size_t> b_batch(other.shape_.begin(), other.shape_.end() - 2);
        if (a_batch != b_batch) throw std::invalid_argument("batched matmul requires identical batch dimensions");
        std::vector<std::size_t> out_shape = a_batch;
        out_shape.push_back(a_m);
        out_shape.push_back(b_n);
        out = Tensor(out_shape);
        const auto batch_count = flat_batch_count(a_batch);
        const auto a_matrix_size = a_m * a_k;
        const auto b_matrix_size = a_k * b_n;
        const auto out_matrix_size = a_m * b_n;
        for (std::size_t batch = 0; batch < batch_count; ++batch) {
            neon::matmul_2d(data_.data() + batch * a_matrix_size,
                            other.data_.data() + batch * b_matrix_size,
                            out.data_.data() + batch * out_matrix_size,
                            a_m, a_k, b_n);
        }
    } else {
        throw std::invalid_argument("Unsupported matmul rank combination");
    }

    if (requires_grad() || other.requires_grad()) {
        auto node = std::make_shared<AutogradNode>();
        node->parents = {*this, other};
        const Tensor a = *this;
        const Tensor b = other;
        node->backward = [a, b](const Tensor& grad) {
            Tensor ga;
            Tensor gb;
            const auto ra2 = a.ndim();
            const auto rb2 = b.ndim();
            if (ra2 == 1 && rb2 == 1) {
                ga = b.mul(grad.item());
                gb = a.mul(grad.item());
            } else if (ra2 == 2 && rb2 == 1) {
                const auto m = a.shape()[0];
                const auto k = a.shape()[1];
                ga = Tensor({m, k});
                gb = Tensor({k});
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t p = 0; p < k; ++p) ga.data()[i * k + p] = grad.data()[i] * b.data()[p];
                }
                for (std::size_t p = 0; p < k; ++p) {
                    float s = 0.0f;
                    for (std::size_t i = 0; i < m; ++i) s += a.data()[i * k + p] * grad.data()[i];
                    gb.data()[p] = s;
                }
            } else if (ra2 == 1 && rb2 == 2) {
                const auto k = a.shape()[0];
                const auto n = b.shape()[1];
                ga = Tensor({k});
                gb = Tensor({k, n});
                for (std::size_t p = 0; p < k; ++p) {
                    float s = 0.0f;
                    for (std::size_t j = 0; j < n; ++j) s += b.data()[p * n + j] * grad.data()[j];
                    ga.data()[p] = s;
                    for (std::size_t j = 0; j < n; ++j) gb.data()[p * n + j] = a.data()[p] * grad.data()[j];
                }
            } else if (ra2 >= 2 && rb2 >= 2) {
                ga = grad.matmul(transpose_last2(b));
                gb = transpose_last2(a).matmul(grad);
            } else {
                throw std::invalid_argument("Unsupported matmul backward rank combination");
            }
            return std::vector<Tensor>{ga, gb};
        };
        out.attach_grad_fn(node);
    }
    return out;
}

bool Tensor::requires_grad() const noexcept { return autograd_ && autograd_->requires_grad; }

void Tensor::set_requires_grad(bool value) {
    if (!autograd_) autograd_ = std::make_shared<AutogradMeta>();
    autograd_->requires_grad = value;
    if (value) {
        autograd_->grad.assign(data_.size(), 0.0f);
    } else {
        autograd_->grad.clear();
        autograd_->grad_fn.reset();
    }
}

bool Tensor::has_grad() const noexcept {
    return autograd_ && !autograd_->grad.empty();
}

Tensor Tensor::grad() const {
    if (!has_grad()) throw std::runtime_error("Tensor has no accumulated gradient");
    return Tensor(shape_, autograd_->grad);
}

void Tensor::zero_grad() {
    if (requires_grad()) autograd_->grad.assign(data_.size(), 0.0f);
}

void Tensor::attach_grad_fn(std::shared_ptr<AutogradNode> node) {
    if (!g_autograd_enabled || !node) return;
    bool any = false;
    for (const auto& parent : node->parents) any = any || parent.requires_grad();
    if (!any) return;
    if (!autograd_) autograd_ = std::make_shared<AutogradMeta>();
    autograd_->requires_grad = true;
    autograd_->grad_fn = std::move(node);
    autograd_->grad.assign(data_.size(), 0.0f);
}

void Tensor::accumulate_grad(const std::shared_ptr<AutogradMeta>& meta, const Tensor& contribution) {
    if (!meta || !meta->requires_grad) return;
    if (meta->grad.size() != contribution.size()) meta->grad.assign(contribution.size(), 0.0f);
    if (meta->grad.empty()) return;
    for (std::size_t i = 0; i < meta->grad.size(); ++i) meta->grad[i] += contribution.data()[i];
}

std::vector<Tensor> Tensor::topo_sort(const Tensor& root) {
    std::vector<Tensor> order;
    std::unordered_set<const AutogradMeta*> visited;
    std::function<void(const Tensor&)> dfs = [&](const Tensor& t) {
        auto meta = t.autograd_;
        if (!meta || !visited.insert(meta.get()).second) return;
        if (meta->grad_fn) {
            for (const auto& parent : meta->grad_fn->parents) dfs(parent);
        }
        order.push_back(t);
    };
    dfs(root);
    return order;
}

void Tensor::backward() {
    if (data_.size() != 1) {
        throw std::invalid_argument("backward() requires a scalar output; provide an explicit gradient for non-scalars");
    }
    Tensor grad_tensor = Tensor({}, {1.0f});
    backward(grad_tensor);
}

void Tensor::backward(const Tensor& grad) {
    if (grad.shape() != shape_) throw std::invalid_argument("backward gradient shape must match output shape");
    if (!requires_grad()) throw std::runtime_error("cannot call backward on a tensor that does not require gradients");

    auto order = topo_sort(*this);
    NoGradGuard no_grad;
    std::unordered_map<AutogradMeta*, Tensor> local_grads;
    local_grads.emplace(autograd_.get(), grad);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Tensor current = *it;
        if (!current.autograd_) continue;
        auto local_it = local_grads.find(current.autograd_.get());
        if (local_it == local_grads.end()) continue;
        if (!current.autograd_->grad_fn) continue;

        const Tensor current_grad = local_it->second;
        const auto& node = current.autograd_->grad_fn;
        const auto contributions = node->backward(current_grad);
        if (contributions.size() != node->parents.size()) {
            throw std::runtime_error("autograd backward returned wrong number of gradients");
        }
        for (std::size_t i = 0; i < node->parents.size(); ++i) {
            auto parent_meta = node->parents[i].autograd_;
            if (!parent_meta || !parent_meta->requires_grad) continue;
            auto found = local_grads.find(parent_meta.get());
            if (found == local_grads.end()) {
                local_grads.emplace(parent_meta.get(), contributions[i]);
            } else {
                ensure_same_shape(found->second, contributions[i], "autograd gradient accumulation");
                for (std::size_t j = 0; j < found->second.data().size(); ++j) {
                    found->second.data()[j] += contributions[i].data()[j];
                }
            }
        }
    }

    for (auto& [meta_ptr, local] : local_grads) {
        if (!meta_ptr || !meta_ptr->requires_grad) continue;
        if (meta_ptr->grad.size() != local.size()) meta_ptr->grad.assign(local.size(), 0.0f);
        for (std::size_t i = 0; i < local.size(); ++i) meta_ptr->grad[i] += local.data()[i];
    }
}

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) oss << ", ";
        oss << shape_[i];
    }
    oss << "], size=" << data_.size() << ", requires_grad=" << (requires_grad() ? "true" : "false") << ")";
    return oss.str();
}

} // namespace orso

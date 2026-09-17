#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace orso {

Tensor::Tensor() : shape_{}, strides_{}, data_{0.0f} {}

Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape), data_(checked_numel(shape), 0.0f) {
    validate_shape(shape_);
    rebuild_strides();
}

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data) {
    validate_shape(shape_);
    const auto expected = checked_numel(shape_);
    if (expected != data_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape");
    }
    rebuild_strides();
}

Tensor Tensor::zeros(const std::vector<std::size_t>& shape) {
    return Tensor(shape);
}

Tensor Tensor::ones(const std::vector<std::size_t>& shape) {
    return full(shape, 1.0f);
}

Tensor Tensor::full(const std::vector<std::size_t>& shape, float value) {
    Tensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), value);
    return t;
}

void Tensor::validate_shape(const std::vector<std::size_t>& shape) {
    // Empty shape is a scalar. A zero-sized dimension is allowed and yields an empty tensor.
    for (const auto dim : shape) {
        (void)dim;
    }
}

std::size_t Tensor::checked_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }

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
    if (shape.empty()) {
        return strides;
    }

    for (std::size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

void Tensor::rebuild_strides() {
    strides_ = make_contiguous_strides(shape_);
}

std::size_t Tensor::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index rank does not match tensor rank");
    }

    std::size_t result = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        result += indices[i] * strides_[i];
    }
    return result;
}

float Tensor::item() const {
    if (data_.size() != 1) {
        throw std::invalid_argument("item() requires a tensor with exactly one element");
    }
    return data_[0];
}

float Tensor::get(const std::vector<std::size_t>& indices) const {
    return data_[offset(indices)];
}

void Tensor::set(const std::vector<std::size_t>& indices, float value) {
    data_[offset(indices)] = value;
}

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
    const auto expected = checked_numel(new_shape);
    if (expected != data_.size()) {
        throw std::invalid_argument("reshape cannot change the number of elements");
    }
    return Tensor(new_shape, data_);
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
    for (std::size_t i = 0; i < axes.size(); ++i) {
        new_shape[i] = shape_[axes[i]];
    }

    Tensor out(new_shape);
    if (data_.empty()) {
        return out;
    }

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
    return out;
}

Tensor Tensor::elementwise_same_shape(const Tensor& other, char op) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Elementwise operation requires equal tensor shapes");
    }

    Tensor out(shape_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        switch (op) {
            case '+': out.data_[i] = data_[i] + other.data_[i]; break;
            case '-': out.data_[i] = data_[i] - other.data_[i]; break;
            case '*': out.data_[i] = data_[i] * other.data_[i]; break;
            case '/':
                if (other.data_[i] == 0.0f) {
                    throw std::domain_error("Tensor division by zero");
                }
                out.data_[i] = data_[i] / other.data_[i];
                break;
            default: throw std::invalid_argument("Unknown elementwise operation");
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const { return elementwise_same_shape(other, '+'); }
Tensor Tensor::sub(const Tensor& other) const { return elementwise_same_shape(other, '-'); }
Tensor Tensor::mul(const Tensor& other) const { return elementwise_same_shape(other, '*'); }
Tensor Tensor::div(const Tensor& other) const { return elementwise_same_shape(other, '/'); }

Tensor Tensor::mul(float scalar) const {
    Tensor out(shape_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        out.data_[i] = data_[i] * scalar;
    }
    return out;
}

Tensor Tensor::div(float scalar) const {
    if (scalar == 0.0f) {
        throw std::domain_error("Tensor division by zero");
    }
    return mul(1.0f / scalar);
}

Tensor Tensor::neg() const {
    return mul(-1.0f);
}

std::size_t Tensor::flat_batch_count(const std::vector<std::size_t>& batch_shape) {
    return checked_numel(batch_shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    const auto ra = ndim();
    const auto rb = other.ndim();
    if (ra == 0 || rb == 0) {
        throw std::invalid_argument("matmul requires tensors with rank >= 1");
    }

    // Vector x Vector -> scalar tensor (shape []).
    if (ra == 1 && rb == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("matmul dimension mismatch");
        }
        Tensor out(std::vector<std::size_t>{});
        float sum = 0.0f;
        for (std::size_t k = 0; k < shape_[0]; ++k) {
            sum += data_[k] * other.data_[k];
        }
        out.data_[0] = sum;
        return out;
    }

    // Matrix x Vector -> Vector.
    if (ra == 2 && rb == 1) {
        const auto m = shape_[0];
        const auto k = shape_[1];
        if (k != other.shape_[0]) {
            throw std::invalid_argument("matmul dimension mismatch");
        }
        Tensor out({m});
        for (std::size_t i = 0; i < m; ++i) {
            float sum = 0.0f;
            const auto base = i * k;
            for (std::size_t j = 0; j < k; ++j) {
                sum += data_[base + j] * other.data_[j];
            }
            out.data_[i] = sum;
        }
        return out;
    }

    // Vector x Matrix -> Vector.
    if (ra == 1 && rb == 2) {
        const auto k = shape_[0];
        const auto n = other.shape_[1];
        if (k != other.shape_[0]) {
            throw std::invalid_argument("matmul dimension mismatch");
        }
        Tensor out({n});
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < k; ++i) {
                sum += data_[i] * other.data_[i * n + j];
            }
            out.data_[j] = sum;
        }
        return out;
    }

    // General batched matrix multiplication with identical batch dimensions.
    // Supports rank >= 2 on both operands. Output preserves the common batch prefix.
    if (ra >= 2 && rb >= 2) {
        const auto a_m = shape_[ra - 2];
        const auto a_k = shape_[ra - 1];
        const auto b_k = other.shape_[rb - 2];
        const auto b_n = other.shape_[rb - 1];
        if (a_k != b_k) {
            throw std::invalid_argument("matmul inner dimensions mismatch");
        }

        const std::vector<std::size_t> a_batch(shape_.begin(), shape_.end() - 2);
        const std::vector<std::size_t> b_batch(other.shape_.begin(), other.shape_.end() - 2);
        if (a_batch != b_batch) {
            throw std::invalid_argument("batched matmul requires identical batch dimensions");
        }

        std::vector<std::size_t> out_shape = a_batch;
        out_shape.push_back(a_m);
        out_shape.push_back(b_n);
        Tensor out(out_shape);

        const auto batch_count = flat_batch_count(a_batch);
        const auto a_matrix_size = a_m * a_k;
        const auto b_matrix_size = a_k * b_n;
        const auto out_matrix_size = a_m * b_n;

        for (std::size_t batch = 0; batch < batch_count; ++batch) {
            const auto a_base = batch * a_matrix_size;
            const auto b_base = batch * b_matrix_size;
            const auto o_base = batch * out_matrix_size;
            for (std::size_t i = 0; i < a_m; ++i) {
                for (std::size_t j = 0; j < b_n; ++j) {
                    float sum = 0.0f;
                    for (std::size_t k = 0; k < a_k; ++k) {
                        sum += data_[a_base + i * a_k + k] * other.data_[b_base + k * b_n + j];
                    }
                    out.data_[o_base + i * b_n + j] = sum;
                }
            }
        }
        return out;
    }

    throw std::invalid_argument("Unsupported matmul rank combination");
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) oss << ", ";
        oss << shape_[i];
    }
    oss << "], size=" << data_.size() << ")";
    return oss.str();
}

} // namespace orso

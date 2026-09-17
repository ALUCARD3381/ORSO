#include "orso/tensor.hpp"
#include "orso/neon_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace orso {
namespace {

std::size_t numel(const Shape& shape) {
    if (shape.empty()) return 1;
    std::size_t n = 1;
    for (std::size_t d : shape) {
        if (d == 0) return 0;
        if (n > std::numeric_limits<std::size_t>::max() / d) throw std::overflow_error("Tensor size overflow");
        n *= d;
    }
    return n;
}

std::vector<std::size_t> strides_for(const Shape& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    if (shape.empty()) return strides;
    for (std::size_t i = shape.size(); i-- > 1;) strides[i - 1] = strides[i] * shape[i];
    return strides;
}

void validate_index(const Shape& shape, const std::vector<std::size_t>& idx) {
    if (idx.size() != shape.size()) throw std::invalid_argument("Index rank mismatch");
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (idx[i] >= shape[i]) throw std::out_of_range("Tensor index out of range");
}

std::size_t flat_index(const Shape& shape, const std::vector<std::size_t>& idx) {
    validate_index(shape, idx);
    const auto strides = strides_for(shape);
    std::size_t out = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) out += idx[i] * strides[i];
    return out;
}

void check_same_numel(const Shape& a, const Shape& b) {
    if (numel(a) != numel(b)) throw std::invalid_argument("Reshape changes tensor size");
}

std::shared_ptr<BackwardNode> node_for(const std::vector<Tensor>& parents,
                                       std::function<void(const std::vector<float>&)> fn) {
    auto node = std::make_shared<BackwardNode>();
    for (const Tensor& p : parents) node->parents.push_back(p.impl());
    node->backward = std::move(fn);
    return node;
}

Tensor make_result(const Shape& shape, std::vector<float> data, bool req, std::shared_ptr<BackwardNode> node = nullptr) {
    auto impl = std::make_shared<TensorImpl>();
    impl->shape = shape;
    impl->data = std::move(data);
    impl->requires_grad = req;
    impl->grad_fn = std::move(node);
    if (req) impl->grad.assign(impl->data.size(), 0.0f);
    return Tensor(std::move(impl));
}

void add_same_shape_grad(const std::shared_ptr<TensorImpl>& target, const std::vector<float>& g, float scale = 1.0f) {
    if (!target || !target->requires_grad) return;
    if (target->grad.empty()) target->grad.assign(target->data.size(), 0.0f);
    for (std::size_t i = 0; i < g.size(); ++i) target->grad[i] += scale * g[i];
}

} // namespace

Tensor::Tensor() : impl_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(const Shape& shape, float fill, bool requires_grad) : impl_(std::make_shared<TensorImpl>()) {
    impl_->shape = shape;
    impl_->data.assign(numel(shape), fill);
    impl_->requires_grad = requires_grad;
    if (requires_grad) impl_->grad.assign(impl_->data.size(), 0.0f);
}

Tensor::Tensor(const Shape& shape, const std::vector<float>& values, bool requires_grad) : impl_(std::make_shared<TensorImpl>()) {
    if (numel(shape) != values.size()) throw std::invalid_argument("Data size does not match shape");
    impl_->shape = shape;
    impl_->data = values;
    impl_->requires_grad = requires_grad;
    if (requires_grad) impl_->grad.assign(impl_->data.size(), 0.0f);
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor Tensor::zeros(const Shape& shape, bool requires_grad) { return Tensor(shape, 0.0f, requires_grad); }
Tensor Tensor::ones(const Shape& shape, bool requires_grad) { return Tensor(shape, 1.0f, requires_grad); }
Tensor Tensor::full(const Shape& shape, float value, bool requires_grad) { return Tensor(shape, value, requires_grad); }

Tensor Tensor::random_normal(const Shape& shape, float mean, float stddev,
                             unsigned long long seed, bool requires_grad) {
    Tensor t(shape, 0.0f, requires_grad);
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (float& v : t.impl_->data) v = dist(rng);
    return t;
}

const Shape& Tensor::shape() const { return impl_->shape; }
std::size_t Tensor::ndim() const { return impl_->shape.size(); }
std::size_t Tensor::size() const { return impl_->data.size(); }
bool Tensor::requires_grad() const { return impl_->requires_grad; }
void Tensor::set_requires_grad(bool value) {
    impl_->requires_grad = value;
    if (value && impl_->grad.empty()) impl_->grad.assign(impl_->data.size(), 0.0f);
    if (!value) { impl_->grad.clear(); impl_->grad_fn.reset(); }
}
const std::vector<float>& Tensor::data() const { return impl_->data; }
std::vector<float>& Tensor::mutable_data() { return impl_->data; }

float Tensor::item() const {
    if (impl_->data.size() != 1) throw std::invalid_argument("item() requires a single-element tensor");
    return impl_->data[0];
}

float Tensor::get(const std::vector<std::size_t>& index) const { return impl_->data[flat_index(impl_->shape, index)]; }
void Tensor::set(const std::vector<std::size_t>& index, float value) { impl_->data[flat_index(impl_->shape, index)] = value; }
const std::vector<float>& Tensor::grad() const { return impl_->grad; }
void Tensor::zero_grad() { if (impl_->requires_grad) std::fill(impl_->grad.begin(), impl_->grad.end(), 0.0f); }

void Tensor::backward(const Tensor* grad_output) {
    if (!impl_->requires_grad && !impl_->grad_fn) return;
    std::vector<float> seed;
    if (grad_output) {
        if (grad_output->shape() != impl_->shape) throw std::invalid_argument("Gradient shape mismatch in backward()");
        seed = grad_output->data();
    } else {
        if (impl_->data.size() != 1) throw std::invalid_argument("backward() without gradient requires scalar tensor");
        seed.assign(1, 1.0f);
    }

    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::unordered_set<TensorImpl*> seen;
    std::function<void(const std::shared_ptr<TensorImpl>&)> visit = [&](const std::shared_ptr<TensorImpl>& node) {
        if (!node || !seen.insert(node.get()).second) return;
        if (node->grad_fn) {
            for (const auto& p : node->grad_fn->parents) visit(p);
        }
        topo.push_back(node);
    };
    visit(impl_);

    if (impl_->grad.empty()) impl_->grad.assign(impl_->data.size(), 0.0f);
    for (std::size_t i = 0; i < seed.size(); ++i) impl_->grad[i] += seed[i];

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->grad_fn) (*it)->grad_fn->backward((*it)->grad);
    }
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    check_same_numel(impl_->shape, new_shape);
    auto req = impl_->requires_grad;
    auto result = make_result(new_shape, impl_->data, req);
    if (req) {
        result.impl()->grad_fn = node_for({*this}, [parent = impl_, old_shape = impl_->shape](const std::vector<float>& gout) {
            (void)old_shape;
            add_same_shape_grad(parent, gout);
        });
    }
    return result;
}

Tensor Tensor::transpose(const std::vector<std::size_t>& permutation) const {
    if (permutation.size() != impl_->shape.size()) throw std::invalid_argument("Permutation rank mismatch");
    std::vector<bool> used(permutation.size(), false);
    for (auto p : permutation) {
        if (p >= permutation.size() || used[p]) throw std::invalid_argument("Invalid transpose permutation");
        used[p] = true;
    }
    Shape out_shape(permutation.size());
    for (std::size_t i = 0; i < permutation.size(); ++i) out_shape[i] = impl_->shape[permutation[i]];
    const auto in_strides = strides_for(impl_->shape);
    const auto out_strides = strides_for(out_shape);
    std::vector<float> out(impl_->data.size());
    for (std::size_t flat = 0; flat < out.size(); ++flat) {
        std::size_t rem = flat;
        std::vector<std::size_t> out_idx(out_shape.size(), 0);
        for (std::size_t i = 0; i < out_shape.size(); ++i) {
            out_idx[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }
        std::size_t in_flat = 0;
        for (std::size_t i = 0; i < permutation.size(); ++i) in_flat += out_idx[i] * in_strides[permutation[i]];
        out[flat] = impl_->data[in_flat];
    }

    auto req = impl_->requires_grad;
    auto result = make_result(out_shape, std::move(out), req);
    if (req) {
        result.impl()->grad_fn = node_for({*this}, [parent = impl_, permutation, out_shape](const std::vector<float>& gout) {
            if (!parent->requires_grad) return;
            const auto in_strides = strides_for(parent->shape);
            const auto out_strides = strides_for(out_shape);
            std::vector<float> gin(parent->data.size(), 0.0f);
            std::vector<std::size_t> in_idx(parent->shape.size(), 0);
            std::vector<std::size_t> out_idx(out_shape.size(), 0);
            for (std::size_t in_flat = 0; in_flat < gin.size(); ++in_flat) {
                std::size_t rem = in_flat;
                for (std::size_t i = 0; i < parent->shape.size(); ++i) {
                    in_idx[i] = rem / in_strides[i];
                    rem %= in_strides[i];
                }
                for (std::size_t i = 0; i < permutation.size(); ++i) out_idx[i] = in_idx[permutation[i]];
                std::size_t out_flat = 0;
                for (std::size_t i = 0; i < out_idx.size(); ++i) out_flat += out_idx[i] * out_strides[i];
                gin[in_flat] += gout[out_flat];
            }
            add_same_shape_grad(parent, gin);
        });
    }
    return result;
}

Tensor Tensor::sum() const {
    float s = std::accumulate(impl_->data.begin(), impl_->data.end(), 0.0f);
    auto req = impl_->requires_grad;
    auto result = make_result({1}, {s}, req);
    if (req) result.impl()->grad_fn = node_for({*this}, [parent = impl_](const std::vector<float>& gout) {
        if (!parent->requires_grad) return;
        add_same_shape_grad(parent, std::vector<float>(parent->data.size(), gout[0]));
    });
    return result;
}

Tensor Tensor::mean() const {
    if (impl_->data.empty()) throw std::invalid_argument("mean() of empty tensor");
    const float inv = 1.0f / static_cast<float>(impl_->data.size());
    float s = std::accumulate(impl_->data.begin(), impl_->data.end(), 0.0f) * inv;
    auto req = impl_->requires_grad;
    auto result = make_result({1}, {s}, req);
    if (req) result.impl()->grad_fn = node_for({*this}, [parent = impl_, inv](const std::vector<float>& gout) {
        if (!parent->requires_grad) return;
        add_same_shape_grad(parent, std::vector<float>(parent->data.size(), gout[0] * inv));
    });
    return result;
}

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape().size(); ++i) { if (i) oss << ", "; oss << shape()[i]; }
    oss << "], requires_grad=" << (requires_grad() ? "true" : "false") << ", data=[";
    const std::size_t preview = std::min<std::size_t>(data().size(), 8);
    for (std::size_t i = 0; i < preview; ++i) { if (i) oss << ", "; oss << std::fixed << std::setprecision(5) << data()[i]; }
    if (data().size() > preview) oss << ", ...";
    oss << "])";
    return oss.str();
}

void ensure_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) throw std::invalid_argument(std::string(op) + ": shapes must match");
}

void accumulate_grad(const std::shared_ptr<TensorImpl>& target, const std::vector<float>& grad) {
    add_same_shape_grad(target, grad);
}

namespace {

template <typename Forward, typename Backward>
Tensor elementwise_binary(const Tensor& a, const Tensor& b, Forward fwd, Backward bwd) {
    ensure_same_shape(a, b, "elementwise op");
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = fwd(a.data()[i], b.data()[i]);
    const bool req = a.requires_grad() || b.requires_grad();
    auto result = make_result(a.shape(), std::move(out), req);
    if (req) {
        result.impl()->grad_fn = node_for({a, b}, [ia = a.impl(), ib = b.impl(), op_bwd = std::move(bwd)](const std::vector<float>& gout) mutable {
            std::vector<float> ga(ia->data.size(), 0.0f), gb(ib->data.size(), 0.0f);
            for (std::size_t i = 0; i < gout.size(); ++i) op_bwd(ia->data[i], ib->data[i], gout[i], ga[i], gb[i]);
            add_same_shape_grad(ia, ga);
            add_same_shape_grad(ib, gb);
        });
    }
    return result;
}

template <typename Forward, typename Backward>
Tensor elementwise_unary(const Tensor& a, Forward fwd, Backward bwd) {
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = fwd(a.data()[i]);
    auto req = a.requires_grad();
    auto result = make_result(a.shape(), std::move(out), req);
    if (req) result.impl()->grad_fn = node_for({a}, [ia = a.impl(), op_bwd = std::move(bwd)](const std::vector<float>& gout) mutable {
        std::vector<float> ga(ia->data.size(), 0.0f);
        for (std::size_t i = 0; i < gout.size(); ++i) ga[i] = op_bwd(ia->data[i], gout[i]);
        add_same_shape_grad(ia, ga);
    });
    return result;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b, [](float x, float y) { return x + y; },
                              [](float, float, float g, float& ga, float& gb) { ga = g; gb = g; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b, [](float x, float y) { return x - y; },
                              [](float, float, float g, float& ga, float& gb) { ga = g; gb = -g; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
    ensure_same_shape(a, b, "mul");
    std::vector<float> out(a.size()); kernels::mul_f32(a.data().data(), b.data().data(), out.data(), out.size());
    auto req = a.requires_grad() || b.requires_grad();
    auto result = make_result(a.shape(), std::move(out), req);
    if (req) result.impl()->grad_fn = node_for({a, b}, [ia=a.impl(), ib=b.impl()](const std::vector<float>& gout) {
        std::vector<float> ga(ia->data.size()), gb(ib->data.size());
        for (std::size_t i=0;i<gout.size();++i) { ga[i]=gout[i]*ib->data[i]; gb[i]=gout[i]*ia->data[i]; }
        add_same_shape_grad(ia,ga); add_same_shape_grad(ib,gb);
    });
    return result;
}
Tensor div(const Tensor& a, const Tensor& b) {
    ensure_same_shape(a, b, "div");
    std::vector<float> out(a.size());
    for (std::size_t i=0;i<out.size();++i) out[i]=a.data()[i]/b.data()[i];
    auto req = a.requires_grad() || b.requires_grad();
    auto result = make_result(a.shape(), std::move(out), req);
    if (req) result.impl()->grad_fn = node_for({a,b}, [ia=a.impl(),ib=b.impl()](const std::vector<float>& gout) {
        std::vector<float> ga(ia->data.size()), gb(ib->data.size());
        for (std::size_t i=0;i<gout.size();++i) {
            ga[i]=gout[i]/ib->data[i];
            gb[i]=-gout[i]*ia->data[i]/(ib->data[i]*ib->data[i]);
        }
        add_same_shape_grad(ia,ga); add_same_shape_grad(ib,gb);
    });
    return result;
}
Tensor add(const Tensor& a, float scalar) { return elementwise_unary(a,[scalar](float x){return x+scalar;},[](float,float g){return g;}); }
Tensor sub(const Tensor& a, float scalar) { return elementwise_unary(a,[scalar](float x){return x-scalar;},[](float,float g){return g;}); }
Tensor mul(const Tensor& a, float scalar) { return elementwise_unary(a,[scalar](float x){return x*scalar;},[scalar](float,float g){return g*scalar;}); }
Tensor div(const Tensor& a, float scalar) { return elementwise_unary(a,[scalar](float x){return x/scalar;},[scalar](float,float g){return g/scalar;}); }
Tensor neg(const Tensor& a) { return elementwise_unary(a,[](float x){return -x;},[](float,float g){return -g;}); }
Tensor exp(const Tensor& a) { return elementwise_unary(a,[](float x){return std::exp(x);},[](float x,float g){return g*std::exp(x);}); }
Tensor log(const Tensor& a) { return elementwise_unary(a,[](float x){return std::log(x);},[](float x,float g){return g/x;}); }
Tensor sqrt(const Tensor& a) { return elementwise_unary(a,[](float x){return std::sqrt(x);},[](float x,float g){return g/(2.0f*std::sqrt(x));}); }
Tensor silu(const Tensor& a) {
    return elementwise_unary(a, [](float x){ const float s=1.0f/(1.0f+std::exp(-x)); return x*s; },
                             [](float x,float g){ const float s=1.0f/(1.0f+std::exp(-x)); return g*(s+x*s*(1.0f-s)); });
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 0 || b.ndim() == 0) throw std::invalid_argument("matmul requires rank >= 1");
    if (a.ndim() == 1 && b.ndim() == 1) {
        if (a.shape()[0] != b.shape()[0]) throw std::invalid_argument("matmul vector sizes mismatch");
        float s = kernels::dot_f32(a.data().data(), b.data().data(), a.size());
        const bool req = a.requires_grad() || b.requires_grad();
        auto result = make_result({1}, {s}, req);
        if (req) result.impl()->grad_fn = node_for({a,b}, [ia=a.impl(), ib=b.impl()](const std::vector<float>& gout) {
            if (ia->requires_grad) { std::vector<float> g(ia->data.size()); for(size_t i=0;i<g.size();++i) g[i]=gout[0]*ib->data[i]; add_same_shape_grad(ia,g); }
            if (ib->requires_grad) { std::vector<float> g(ib->data.size()); for(size_t i=0;i<g.size();++i) g[i]=gout[0]*ia->data[i]; add_same_shape_grad(ib,g); }
        });
        return result;
    }
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul supports vector-vector or tensors with rank >= 2");
    }
    const std::size_t M = a.shape()[a.ndim()-2];
    const std::size_t K = a.shape()[a.ndim()-1];
    const std::size_t Kb = b.shape()[b.ndim()-2];
    const std::size_t N = b.shape()[b.ndim()-1];
    if (K != Kb) throw std::invalid_argument("matmul inner dimensions mismatch");

    Shape a_batch(a.shape().begin(), a.shape().end()-2);
    Shape b_batch(b.shape().begin(), b.shape().end()-2);
    bool b_broadcast = false;
    if (b_batch.empty()) b_broadcast = true;
    else if (a_batch != b_batch) throw std::invalid_argument("matmul batch dimensions mismatch");

    Shape out_shape = a_batch;
    out_shape.push_back(M); out_shape.push_back(N);
    std::vector<float> out(numel(out_shape), 0.0f);
    const std::size_t batch_count = a_batch.empty() ? 1 : numel(a_batch);
    for (std::size_t batch=0; batch<batch_count; ++batch) {
        const float* ap = a.data().data() + batch*M*K;
        const float* bp = b.data().data() + (b_broadcast ? 0 : batch*K*N);
        float* cp = out.data() + batch*M*N;
        kernels::matmul_f32(ap, bp, cp, M, K, N);
    }
    const bool req = a.requires_grad() || b.requires_grad();
    auto result = make_result(out_shape, std::move(out), req);
    if (req) result.impl()->grad_fn = node_for({a,b}, [ia=a.impl(), ib=b.impl(), out_shape, a_batch, b_broadcast, M,K,N,b_batch](const std::vector<float>& gout) {
        const std::size_t batch_count_local = a_batch.empty() ? 1 : numel(a_batch);
        if (ia->requires_grad) {
            std::vector<float> ga(ia->data.size(),0.0f);
            for(size_t batch=0;batch<batch_count_local;++batch){
                const float* bp=ib->data.data()+(b_broadcast?0:batch*K*N);
                const float* gp=gout.data()+batch*M*N;
                float* apg=ga.data()+batch*M*K;
                for(size_t i=0;i<M;++i) for(size_t k=0;k<K;++k){ float s=0; for(size_t j=0;j<N;++j) s+=gp[i*N+j]*bp[k*N+j]; apg[i*K+k]+=s; }
            }
            add_same_shape_grad(ia,ga);
        }
        if (ib->requires_grad) {
            std::vector<float> gb(ib->data.size(),0.0f);
            for(size_t batch=0;batch<batch_count_local;++batch){
                const float* ap=ia->data.data()+batch*M*K;
                const float* gp=gout.data()+batch*M*N;
                const size_t bbase=b_broadcast?0:batch*K*N;
                for(size_t k=0;k<K;++k) for(size_t j=0;j<N;++j){ float s=0; for(size_t i=0;i<M;++i) s+=ap[i*K+k]*gp[i*N+j]; gb[bbase+k*N+j]+=s; }
            }
            add_same_shape_grad(ib,gb);
        }
    });
    return result;
}

Tensor softmax(const Tensor& x, int axis) {
    if (x.ndim() == 0) throw std::invalid_argument("softmax requires rank >= 1");
    int ax = axis < 0 ? static_cast<int>(x.ndim()) + axis : axis;
    if (ax < 0 || ax >= static_cast<int>(x.ndim())) throw std::invalid_argument("softmax axis out of range");
    Shape shape = x.shape();
    const std::size_t axis_size = shape[ax];
    const std::size_t outer = [&](){ std::size_t n=1; for(int i=0;i<ax;++i)n*=shape[i]; return n; }();
    const std::size_t inner = [&](){ std::size_t n=1; for(size_t i=ax+1;i<shape.size();++i)n*=shape[i]; return n; }();
    std::vector<float> out(x.size());
    for(std::size_t o=0;o<outer;++o){
        for(std::size_t i=0;i<inner;++i){
            float maxv=-std::numeric_limits<float>::infinity();
            for(std::size_t a=0;a<axis_size;++a) maxv=std::max(maxv,x.data()[(o*axis_size+a)*inner+i]);
            float sumv=0;
            for(std::size_t a=0;a<axis_size;++a){ float e=std::exp(x.data()[(o*axis_size+a)*inner+i]-maxv); out[(o*axis_size+a)*inner+i]=e; sumv+=e; }
            for(std::size_t a=0;a<axis_size;++a) out[(o*axis_size+a)*inner+i]/=sumv;
        }
    }
    auto req=x.requires_grad();
    auto result=make_result(shape,std::move(out),req);
    if(req) result.impl()->grad_fn=node_for({x},[ix=x.impl(), shape, axis_size, outer, inner](const std::vector<float>& gout){
        std::vector<float> gx(ix->data.size(),0.0f);
        const auto& y = ix; // input only; output is reconstructed below from input for stable backward.
        (void)y;
        // Recompute softmax probabilities from the saved input.
        for(std::size_t o=0;o<outer;++o) for(std::size_t i=0;i<inner;++i){
            float maxv=-std::numeric_limits<float>::infinity();
            for(std::size_t a=0;a<axis_size;++a) maxv=std::max(maxv,ix->data[(o*axis_size+a)*inner+i]);
            float sumv=0; std::vector<float> probs(axis_size);
            for(std::size_t a=0;a<axis_size;++a){ float e=std::exp(ix->data[(o*axis_size+a)*inner+i]-maxv); probs[a]=e; sumv+=e; }
            for(float& p:probs)p/=sumv;
            float dot=0; for(std::size_t a=0;a<axis_size;++a) dot+=gout[(o*axis_size+a)*inner+i]*probs[a];
            for(std::size_t a=0;a<axis_size;++a) gx[(o*axis_size+a)*inner+i]+=probs[a]*(gout[(o*axis_size+a)*inner+i]-dot);
        }
        add_same_shape_grad(ix,gx);
    });
    return result;
}

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.ndim()<1 || weight.ndim()!=1 || weight.shape()[0]!=x.shape().back()) throw std::invalid_argument("rmsnorm shape mismatch");
    const std::size_t d=x.shape().back();
    const std::size_t rows=x.size()/d;
    std::vector<float> out(x.size());
    std::vector<float> inv(rows);
    for(std::size_t r=0;r<rows;++r){
        float mean_sq=0; for(std::size_t j=0;j<d;++j){float v=x.data()[r*d+j]; mean_sq+=v*v;} mean_sq/=static_cast<float>(d);
        inv[r]=1.0f/std::sqrt(mean_sq+eps);
        for(std::size_t j=0;j<d;++j) out[r*d+j]=x.data()[r*d+j]*inv[r]*weight.data()[j];
    }
    const bool req=x.requires_grad()||weight.requires_grad();
    auto result=make_result(x.shape(),std::move(out),req);
    if(req) result.impl()->grad_fn=node_for({x,weight},[ix=x.impl(),iw=weight.impl(),d,rows,eps](const std::vector<float>& gout){
        std::vector<float> gx(ix->data.size(),0.0f), gw(iw->data.size(),0.0f);
        for(std::size_t r=0;r<rows;++r){
            float mean_sq=0; for(size_t j=0;j<d;++j){float v=ix->data[r*d+j];mean_sq+=v*v;} mean_sq/=static_cast<float>(d);
            float inv=1.0f/std::sqrt(mean_sq+eps);
            float S=0; for(size_t j=0;j<d;++j) S+=gout[r*d+j]*iw->data[j]*ix->data[r*d+j];
            for(size_t j=0;j<d;++j){
                float xj=ix->data[r*d+j], gj=gout[r*d+j], wj=iw->data[j];
                gx[r*d+j]+=gj*wj*inv - xj*inv*inv*inv*S/static_cast<float>(d);
                gw[j]+=gj*xj*inv;
            }
        }
        add_same_shape_grad(ix,gx); add_same_shape_grad(iw,gw);
    });
    return result;
}

Tensor rope(const Tensor& x, float theta) {
    if (x.ndim()<2 || (x.shape().back()%2)!=0) throw std::invalid_argument("RoPE requires rank >= 2 and even last dimension");
    const std::size_t d=x.shape().back();
    const std::size_t t=x.shape()[x.ndim()-2];
    const std::size_t rows=x.size()/d;
    const std::size_t prefix_rows=rows/t;
    std::vector<float> out(x.size());
    for(std::size_t r=0;r<rows;++r){
        const std::size_t pos=r%t;
        for(std::size_t j=0;j<d;j+=2){
            const float inv_freq=std::pow(theta,-static_cast<float>(j)/static_cast<float>(d));
            const float angle=static_cast<float>(pos)*inv_freq;
            const float c=std::cos(angle), s=std::sin(angle);
            const float xe=x.data()[r*d+j], xo=x.data()[r*d+j+1];
            out[r*d+j]=xe*c-xo*s; out[r*d+j+1]=xe*s+xo*c;
        }
    }
    (void)prefix_rows;
    auto req=x.requires_grad(); auto result=make_result(x.shape(),std::move(out),req);
    if(req) result.impl()->grad_fn=node_for({x},[ix=x.impl(),d,t,theta](const std::vector<float>& gout){
        std::vector<float> gx(ix->data.size());
        const size_t rows=ix->data.size()/d;
        for(size_t r=0;r<rows;++r){ size_t pos=r%t; for(size_t j=0;j<d;j+=2){
            float inv_freq=std::pow(theta,-static_cast<float>(j)/static_cast<float>(d));
            float angle=static_cast<float>(pos)*inv_freq; float c=std::cos(angle),s=std::sin(angle);
            float ge=gout[r*d+j],go=gout[r*d+j+1]; gx[r*d+j]=ge*c+go*s; gx[r*d+j+1]=-ge*s+go*c;
        }}
        add_same_shape_grad(ix,gx);
    });
    return result;
}

Tensor embedding_lookup(const Tensor& weight, const std::vector<std::vector<int>>& token_ids) {
    if (weight.ndim()!=2) throw std::invalid_argument("Embedding weight must be 2D");
    const std::size_t vocab=weight.shape()[0], d=weight.shape()[1];
    if(token_ids.empty()) throw std::invalid_argument("token_ids cannot be empty");
    const std::size_t rows=token_ids.size(), cols=token_ids[0].size();
    if(cols==0) throw std::invalid_argument("token_ids rows cannot be empty");
    for(const auto& row:token_ids) if(row.size()!=cols) throw std::invalid_argument("token_ids must be rectangular");
    std::vector<float> out(rows*cols*d);
    for(size_t r=0;r<rows;++r) for(size_t c=0;c<cols;++c){ int tok=token_ids[r][c]; if(tok<0||static_cast<size_t>(tok)>=vocab) throw std::out_of_range("token id outside vocabulary"); for(size_t j=0;j<d;++j) out[(r*cols+c)*d+j]=weight.data()[static_cast<size_t>(tok)*d+j]; }
    auto req=weight.requires_grad(); Shape out_shape={rows,cols,d}; auto result=make_result(out_shape,std::move(out),req);
    if(req) result.impl()->grad_fn=node_for({weight},[iw=weight.impl(),token_ids,vocab,d,rows,cols](const std::vector<float>& gout){
        std::vector<float> gw(iw->data.size(),0.0f);
        for(size_t r=0;r<rows;++r) for(size_t c=0;c<cols;++c){ int tok=token_ids[r][c]; size_t base=((r*cols+c)*d); size_t wbase=static_cast<size_t>(tok)*d; for(size_t j=0;j<d;++j) gw[wbase+j]+=gout[base+j]; }
        add_same_shape_grad(iw,gw);
    });
    return result;
}

Tensor cross_entropy(const Tensor& logits, const std::vector<std::vector<int>>& targets, int ignore_index) {
    if (logits.ndim() != 3) throw std::invalid_argument("cross_entropy expects logits=[B,T,V]");
    const std::size_t B = logits.shape()[0];
    const std::size_t T = logits.shape()[1];
    const std::size_t V = logits.shape()[2];
    if (targets.size() != B) throw std::invalid_argument("cross_entropy target batch mismatch");
    for (const auto& row : targets) if (row.size() != T) throw std::invalid_argument("cross_entropy target shape mismatch");

    std::size_t count = 0;
    double total = 0.0;
    std::vector<float> row_probs(V);
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t t = 0; t < T; ++t) {
            const int target = targets[b][t];
            if (target == ignore_index) continue;
            if (target < 0 || static_cast<std::size_t>(target) >= V) throw std::out_of_range("cross_entropy target out of range");
            const std::size_t base = (b * T + t) * V;
            float maxv = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < V; ++j) maxv = std::max(maxv, logits.data()[base + j]);
            double sumexp = 0.0;
            for (std::size_t j = 0; j < V; ++j) {
                row_probs[j] = std::exp(logits.data()[base + j] - maxv);
                sumexp += row_probs[j];
            }
            total += -(static_cast<double>(logits.data()[base + static_cast<std::size_t>(target)]) - static_cast<double>(maxv) - std::log(sumexp));
            ++count;
        }
    }
    if (count == 0) throw std::invalid_argument("cross_entropy has no valid targets");

    const float loss = static_cast<float>(total / static_cast<double>(count));
    const bool req = logits.requires_grad();
    auto result = make_result({1}, {loss}, req);
    if (req) {
        result.impl()->grad_fn = node_for({logits}, [ilogits = logits.impl(), targets, ignore_index, B, T, V, count](const std::vector<float>& gout) {
            std::vector<float> grad(ilogits->data.size(), 0.0f);
            const float scale = gout[0] / static_cast<float>(count);
            std::vector<float> probs(V);
            for (std::size_t b = 0; b < B; ++b) {
                for (std::size_t t = 0; t < T; ++t) {
                    const int target = targets[b][t];
                    if (target == ignore_index) continue;
                    const std::size_t base = (b * T + t) * V;
                    float maxv = -std::numeric_limits<float>::infinity();
                    for (std::size_t j = 0; j < V; ++j) maxv = std::max(maxv, ilogits->data[base + j]);
                    float sumexp = 0.0f;
                    for (std::size_t j = 0; j < V; ++j) {
                        probs[j] = std::exp(ilogits->data[base + j] - maxv);
                        sumexp += probs[j];
                    }
                    for (std::size_t j = 0; j < V; ++j) grad[base + j] += scale * (probs[j] / sumexp);
                    grad[base + static_cast<std::size_t>(target)] -= scale;
                }
            }
            add_same_shape_grad(ilogits, grad);
        });
    }
    return result;
}

} // namespace orso

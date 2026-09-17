#include "orso/transformer.hpp"

#include <cmath>
#include <stdexcept>

namespace orso {

Tensor causal_mask(std::size_t batch, std::size_t heads, std::size_t seq_len) {
    Tensor mask({batch, heads, seq_len, seq_len}, 0.0f, false);
    for (std::size_t b=0;b<batch;++b) for(std::size_t h=0;h<heads;++h)
        for(std::size_t i=0;i<seq_len;++i) for(std::size_t j=0;j<seq_len;++j)
            if(j>i) mask.set({b,h,i,j}, -1.0e9f);
    return mask;
}

Tensor multi_head_attention(const Tensor& x,
                            const Tensor& wq,
                            const Tensor& wk,
                            const Tensor& wv,
                            const Tensor& wo,
                            std::size_t num_heads,
                            bool causal,
                            float rope_theta) {
    if(x.ndim()!=3) throw std::invalid_argument("multi_head_attention expects x=[B,T,D]");
    if(wq.ndim()!=2||wk.ndim()!=2||wv.ndim()!=2||wo.ndim()!=2) throw std::invalid_argument("Attention weights must be 2D");
    const std::size_t B=x.shape()[0], T=x.shape()[1], D=x.shape()[2];
    if(D%num_heads!=0) throw std::invalid_argument("d_model must be divisible by num_heads");
    const std::size_t Dh=D/num_heads;
    if(wq.shape()[0]!=D||wq.shape()[1]!=D||wk.shape()!=wq.shape()||wv.shape()!=wq.shape()||wo.shape()!=wq.shape())
        throw std::invalid_argument("Attention weights must all have shape [D,D]");

    Tensor q = matmul(x, wq);
    Tensor k = matmul(x, wk);
    Tensor v = matmul(x, wv);

    // [B,T,D] -> [B,H,T,Dh]
    q = q.reshape({B,T,num_heads,Dh}).transpose({0,2,1,3});
    k = k.reshape({B,T,num_heads,Dh}).transpose({0,2,1,3});
    v = v.reshape({B,T,num_heads,Dh}).transpose({0,2,1,3});
    q = rope(q, rope_theta);
    k = rope(k, rope_theta);

    Tensor kt = k.transpose({0,1,3,2});
    Tensor scores = div(matmul(q, kt), std::sqrt(static_cast<float>(Dh)));
    if(causal) scores = add(scores, causal_mask(B,num_heads,T));
    Tensor probs = softmax(scores, -1);
    Tensor ctx = matmul(probs, v); // [B,H,T,Dh]
    ctx = ctx.transpose({0,2,1,3}).reshape({B,T,D});
    return matmul(ctx, wo);
}

Tensor swiglu(const Tensor& x,
              const Tensor& gate_weight,
              const Tensor& up_weight,
              const Tensor& down_weight) {
    Tensor gate = matmul(x, gate_weight);
    Tensor up = matmul(x, up_weight);
    Tensor activated = silu(gate);
    Tensor hidden = mul(activated, up);
    return matmul(hidden, down_weight);
}

} // namespace orso

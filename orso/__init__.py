"""ORSO high-level Python package."""

try:
    from orso_core import Tensor, embedding, rope, rmsnorm, multi_head_attention, swiglu, softmax, Model, ModelConfig, AdamW, cross_entropy
except ImportError:
    Tensor = None
    embedding = rope = rmsnorm = multi_head_attention = swiglu = softmax = Model = ModelConfig = AdamW = cross_entropy = None

__all__ = ["Tensor", "embedding", "rope", "rmsnorm", "multi_head_attention", "swiglu", "softmax", "Model", "ModelConfig", "AdamW", "cross_entropy"]

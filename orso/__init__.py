"""ORSO high-level Python package."""

try:
    from orso_core import Tensor, embedding, rope, rmsnorm, multi_head_attention, swiglu, softmax
except ImportError:
    Tensor = None
    embedding = rope = rmsnorm = multi_head_attention = swiglu = softmax = None

__all__ = ["Tensor", "embedding", "rope", "rmsnorm", "multi_head_attention", "swiglu", "softmax"]

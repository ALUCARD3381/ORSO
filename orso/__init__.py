"""ORSO high-level package."""

try:
    from orso_core import Tensor, embedding, rope, rmsnorm, multi_head_attention, swiglu, softmax
except ImportError:
    Tensor = None
    embedding = rope = rmsnorm = multi_head_attention = swiglu = softmax = None

from .model import ModelConfig
from .tokenizer import BPETokenizer
from .dataset import CausalDataset
from .training import CosineScheduler, Trainer
from .checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from .inference import generate_ids, generate_text
from .chat import ChatSession
from .runtime import ORSORuntime

__all__ = [
    "Tensor", "embedding", "rope", "rmsnorm", "multi_head_attention", "swiglu", "softmax",
    "ModelConfig", "BPETokenizer", "CausalDataset", "CosineScheduler", "Trainer",
    "LoadedCheckpoint", "load_checkpoint", "save_checkpoint", "generate_ids", "generate_text",
    "ChatSession", "ORSORuntime",
]

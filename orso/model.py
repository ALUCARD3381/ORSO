"""High-level model helpers for ORSO Phase 5."""

from dataclasses import dataclass

from orso_core import Model as NativeModel


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 768
    d_model: int = 64
    num_heads: int = 8
    hidden_dim: int = 256
    num_layers: int = 6
    context_length: int = 32
    seed: int = 1234

    def build(self) -> NativeModel:
        return NativeModel(
            self.vocab_size,
            self.d_model,
            self.num_heads,
            self.hidden_dim,
            self.num_layers,
            self.context_length,
            self.seed,
        )

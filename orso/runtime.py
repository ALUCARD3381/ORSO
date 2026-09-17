"""Unified ORSO runtime: checkpoint, tokenizer, session, cache and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import load_checkpoint
from .chat import ChatSession
from .inference import generate_ids


@dataclass
class RuntimeStats:
    parameter_count: int
    context_length: int
    vocab_size: int
    trainer_steps: int
    optimizer_steps: int
    kv_cache_length: int


class ORSORuntime:
    """High-level runtime for the complete ORSO pipeline."""

    def __init__(self, state, session: ChatSession | None = None):
        if state.tokenizer is None:
            raise ValueError("checkpoint must contain a tokenizer")
        self.state = state
        self.model = state.model
        self.tokenizer = state.tokenizer
        self.session = session or ChatSession()
        self.reset_kv_cache()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, session: ChatSession | None = None) -> "ORSORuntime":
        return cls(load_checkpoint(path), session=session)

    def reset_kv_cache(self) -> None:
        reset = getattr(self.model, "reset_kv_cache", None)
        if callable(reset):
            reset()

    @property
    def stats(self) -> RuntimeStats:
        cfg = self.model.config
        parameter_count = getattr(self.model, "parameter_count", 0)
        if callable(parameter_count):
            parameter_count = parameter_count()
        kv_len = getattr(self.model, "kv_cache_length", 0)
        if callable(kv_len):
            kv_len = kv_len()
        return RuntimeStats(
            parameter_count=int(parameter_count),
            context_length=int(cfg.context_length),
            vocab_size=int(cfg.vocab_size),
            trainer_steps=int(self.state.trainer_steps),
            optimizer_steps=int(self.state.optimizer.step_count),
            kv_cache_length=int(kv_len),
        )

    def clear_session(self) -> None:
        self.session.clear()
        self.reset_kv_cache()

    def respond(
        self,
        user_text: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        eos_token_id: int | None = None,
    ) -> str:
        prompt = self.session.prompt(user_text)
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        generated = generate_ids(
            self.model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
        reply = self.tokenizer.decode(generated[len(prompt_ids):]).strip()
        self.session.add("user", user_text)
        self.session.add("assistant", reply)
        return reply

    def save_session(self, path: str | Path) -> Path:
        return self.session.save(path)

    def load_session(self, path: str | Path) -> None:
        self.session = ChatSession.load(path)
        self.reset_kv_cache()

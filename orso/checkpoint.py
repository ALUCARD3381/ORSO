"""Versioned ORSO Phase 6 checkpoints: weights, AdamW, tokenizer and resume state."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
from typing import Any
import zipfile

from orso_core import AdamW, Model, ModelConfig

MAGIC = "ORSO_CHECKPOINT"
FORMAT_VERSION = 1


def _config_to_dict(cfg: ModelConfig) -> dict[str, int]:
    return {
        "vocab_size": int(cfg.vocab_size),
        "d_model": int(cfg.d_model),
        "num_heads": int(cfg.num_heads),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "context_length": int(cfg.context_length),
        "seed": int(cfg.seed),
    }


def _config_from_dict(state: dict[str, Any]) -> ModelConfig:
    cfg = ModelConfig()
    for name in ("vocab_size", "d_model", "num_heads", "hidden_dim", "num_layers", "context_length", "seed"):
        if name in state:
            setattr(cfg, name, int(state[name]))
    return cfg


def _pack_arrays(arrays: list[list[float]]) -> bytes:
    out = io.BytesIO()
    for values in arrays:
        raw = array("f", (float(v) for v in values))
        if sys.byteorder != "little":
            raw.byteswap()
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw.tobytes())
    return out.getvalue()


def _unpack_arrays(blob: bytes, expected_count: int) -> list[list[float]]:
    pos = 0
    arrays: list[list[float]] = []
    for _ in range(expected_count):
        if pos + 8 > len(blob):
            raise ValueError("checkpoint array header is truncated")
        (count,) = struct.unpack_from("<Q", blob, pos)
        pos += 8
        byte_count = count * 4
        if pos + byte_count > len(blob):
            raise ValueError("checkpoint array payload is truncated")
        raw = array("f")
        raw.frombytes(blob[pos : pos + byte_count])
        if sys.byteorder != "little":
            raw.byteswap()
        arrays.append(list(raw))
        pos += byte_count
    if pos != len(blob):
        raise ValueError("checkpoint contains trailing binary data")
    return arrays


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass
class LoadedCheckpoint:
    model: Model
    optimizer: AdamW
    tokenizer: Any | None
    trainer_steps: int
    scheduler: Any | None
    metadata: dict[str, Any]

    def build_trainer(self):
        from orso.training import Trainer
        trainer = Trainer(self.model, self.optimizer, self.scheduler)
        trainer.steps = self.trainer_steps
        return trainer


def save_checkpoint(
    path: str | Path,
    model: Model,
    optimizer: AdamW,
    *,
    tokenizer: Any | None = None,
    trainer: Any | None = None,
    scheduler: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a portable .orso checkpoint without external dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if scheduler is None and trainer is not None:
        scheduler = getattr(trainer, "scheduler", None)

    weights = [list(map(float, p)) for p in model.parameter_data()]
    weight_blob = _pack_arrays(weights)
    first = optimizer.first_moment()
    second = optimizer.second_moment()
    optimizer_blob = _pack_arrays([*first, *second])

    cfg = _config_to_dict(model.config)
    scheduler_state = None
    if scheduler is not None:
        scheduler_state = scheduler.state_dict()

    tok_state = tokenizer.to_dict() if tokenizer is not None else None
    meta = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "model_config": cfg,
        "parameter_sizes": [len(x) for x in weights],
        "optimizer": {
            "lr": float(optimizer.lr),
            "beta1": float(optimizer.beta1),
            "beta2": float(optimizer.beta2),
            "eps": float(optimizer.eps),
            "weight_decay": float(optimizer.weight_decay),
            "max_grad_norm": float(optimizer.max_grad_norm),
            "step_count": int(optimizer.step_count),
        },
        "trainer_steps": int(getattr(trainer, "steps", optimizer.step_count)),
        "scheduler": scheduler_state,
        "has_tokenizer": tok_state is not None,
        "metadata": metadata or {},
    }

    meta["payloads"] = {
        "weights.bin": {"sha256": _sha256(weight_blob), "arrays": len(weights)},
        "optimizer.bin": {"sha256": _sha256(optimizer_blob), "arrays": len(first) + len(second)},
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2))
        archive.writestr("weights.bin", weight_blob)
        archive.writestr("optimizer.bin", optimizer_blob)
        if tok_state is not None:
            archive.writestr("tokenizer.json", json.dumps(tok_state, ensure_ascii=False, sort_keys=True))
    return path


def _restore_scheduler(optimizer: AdamW, state: dict[str, Any] | None):
    if state is None:
        return None
    from orso.training import CosineScheduler
    scheduler = CosineScheduler(
        optimizer,
        int(state["total_steps"]),
        int(state["warmup_steps"]),
        float(state["min_lr"]),
    )
    scheduler.load_state_dict(state)
    return scheduler


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load and validate a Phase 6 checkpoint."""
    from orso.tokenizer import BPETokenizer

    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        required = {"metadata.json", "weights.bin", "optimizer.bin"}
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"checkpoint missing files: {sorted(missing)}")
        meta = json.loads(archive.read("metadata.json").decode("utf-8"))
        if meta.get("magic") != MAGIC or int(meta.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unsupported ORSO checkpoint format")
        weight_blob = archive.read("weights.bin")
        optimizer_blob = archive.read("optimizer.bin")
        for name, blob in (("weights.bin", weight_blob), ("optimizer.bin", optimizer_blob)):
            expected = meta.get("payloads", {}).get(name, {}).get("sha256")
            if expected and _sha256(blob) != expected:
                raise ValueError(f"checkpoint integrity check failed for {name}")

        cfg = _config_from_dict(meta["model_config"])
        model = Model(cfg)
        parameter_count = len(model.parameters())
        sizes = list(map(int, meta["parameter_sizes"]))
        if len(sizes) != parameter_count:
            raise ValueError("checkpoint parameter count does not match model configuration")
        weights = _unpack_arrays(weight_blob, parameter_count)
        if [len(x) for x in weights] != sizes:
            raise ValueError("checkpoint parameter sizes do not match metadata")
        model.load_parameter_data(weights)

        opt_meta = meta["optimizer"]
        params = model.parameters()
        optimizer = AdamW(
            params,
            lr=float(opt_meta["lr"]),
            beta1=float(opt_meta["beta1"]),
            beta2=float(opt_meta["beta2"]),
            eps=float(opt_meta["eps"]),
            weight_decay=float(opt_meta["weight_decay"]),
            max_grad_norm=float(opt_meta["max_grad_norm"]),
        )
        half = parameter_count
        states = _unpack_arrays(optimizer_blob, half * 2)
        first, second = states[:half], states[half:]
        optimizer.load_state(first, second, int(opt_meta["step_count"]))

        tokenizer = None
        if meta.get("has_tokenizer"):
            if "tokenizer.json" not in archive.namelist():
                raise ValueError("checkpoint metadata expects tokenizer.json")
            tokenizer = BPETokenizer.from_dict(json.loads(archive.read("tokenizer.json").decode("utf-8")))

    return LoadedCheckpoint(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        trainer_steps=int(meta.get("trainer_steps", optimizer.step_count)),
        scheduler=_restore_scheduler(optimizer, meta.get("scheduler")),
        metadata=meta.get("metadata", {}),
    )

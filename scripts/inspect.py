#!/usr/bin/env python3
"""Inspect an ORSO checkpoint without starting inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orso.checkpoint import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect ORSO checkpoint")
    p.add_argument("checkpoint")
    args = p.parse_args()
    state = load_checkpoint(args.checkpoint)
    cfg = state.model.config
    print(json.dumps({
        "parameter_count": int(state.model.parameter_count()),
        "vocab_size": int(cfg.vocab_size),
        "d_model": int(cfg.d_model),
        "num_heads": int(cfg.num_heads),
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "context_length": int(cfg.context_length),
        "trainer_steps": int(state.trainer_steps),
        "optimizer_steps": int(state.optimizer.step_count),
        "has_tokenizer": state.tokenizer is not None,
        "metadata": state.metadata,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load an ORSO Phase 6 checkpoint and generate text."""

from __future__ import annotations

import argparse

from orso.checkpoint import load_checkpoint
from orso.inference import generate_text


def main() -> None:
    parser = argparse.ArgumentParser(description="ORSO Phase 6 inference")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    state = load_checkpoint(args.checkpoint)
    if state.tokenizer is None:
        raise SystemExit("checkpoint does not contain a tokenizer")
    text = generate_text(
        state.model,
        state.tokenizer,
        args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()

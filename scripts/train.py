#!/usr/bin/env python3
"""Train a small ORSO model end-to-end and write a .orso checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from orso.checkpoint import save_checkpoint
from orso.dataset import CausalDataset
from orso.model import ModelConfig
from orso.tokenizer import BPETokenizer
from orso.training import CosineScheduler, Trainer
from orso_core import AdamW


def main() -> None:
    p = argparse.ArgumentParser(description="ORSO end-to-end trainer")
    p.add_argument("text", help="UTF-8 text corpus")
    p.add_argument("checkpoint", help="output .orso checkpoint")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    tokenizer = BPETokenizer(vocab_size=max(256, args.vocab_size), min_frequency=2).train([text])
    ids = tokenizer.encode(text)
    dataset = CausalDataset(ids, args.context)
    cfg = ModelConfig(
        vocab_size=max(256, tokenizer.size),
        d_model=args.d_model,
        num_heads=args.heads,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        context_length=args.context,
        seed=args.seed,
    )
    model = cfg.build()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, max_grad_norm=1.0)
    scheduler = CosineScheduler(optimizer, total_steps=args.steps, warmup_steps=max(0, min(10, args.steps // 10)))
    trainer = Trainer(model, optimizer, scheduler)

    last = None
    batches = list(dataset.batches(args.batch_size, shuffle=False))
    if not batches:
        raise RuntimeError("dataset produced no batches")
    for step in range(args.steps):
        inputs, targets = batches[step % len(batches)]
        last = trainer.train_batch(inputs, targets)
        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(f"step={int(last['step'])} loss={last['loss']:.6f} lr={last['lr']:.6g} grad={last['grad_norm']:.6f}")

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        tokenizer=tokenizer,
        trainer=trainer,
        scheduler=scheduler,
        metadata={"phase": 7, "script": "scripts/train.py", "text_chars": len(text)},
    )
    print(f"saved={args.checkpoint}")


if __name__ == "__main__":
    main()

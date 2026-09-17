#!/usr/bin/env python3
"""Train a small ORSO model end-to-end and write a .orso checkpoint.

Features:
- train/validation split with periodic evaluation
- periodic "_last" checkpoints for crash recovery, plus a "_best" checkpoint
  tracked by validation loss
- --resume to continue training from any .orso checkpoint, safely extending
  the cosine scheduler to the new --steps target instead of reconstructing it
  from scratch (which would otherwise replay the LR schedule from step 0 and
  leave the optimizer running at the wrong learning rate)
- CSV log of step/loss/val_loss/lr/grad_norm
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from orso.checkpoint import load_checkpoint, save_checkpoint
from orso.dataset import CausalDataset, split_token_ids
from orso.model import ModelConfig
from orso.tokenizer import BPETokenizer
from orso.training import CosineScheduler, Trainer, evaluate
from orso_core import AdamW


def _sibling_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _epoch_batches(dataset: CausalDataset, batch_size: int, shuffle: bool, seed: int):
    """Yields batches forever, reshuffling (with a new seed) at each pass."""
    epoch = 0
    while True:
        yield from dataset.batches(batch_size, shuffle=shuffle, seed=seed + epoch)
        epoch += 1


def _open_log(log_file: str, resume: bool):
    if not log_file:
        return None, None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or not resume
    handle = open(path, "a" if resume and path.exists() else "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if is_new:
        writer.writerow(["step", "loss", "val_loss", "lr", "grad_norm"])
        handle.flush()
    return handle, writer


def main() -> None:
    p = argparse.ArgumentParser(description="ORSO end-to-end trainer")
    p.add_argument("text", help="UTF-8 text corpus")
    p.add_argument("checkpoint", help="output .orso checkpoint (final)")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=100, help="total target training steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-fraction", type=float, default=0.1, help="fraction of tokens held out for validation")
    p.add_argument("--eval-every", type=int, default=0, help="steps between validation passes (0 = auto)")
    p.add_argument("--eval-batch-size", type=int, default=0, help="0 = same as --batch-size")
    p.add_argument("--save-every", type=int, default=0, help="steps between periodic '_last' checkpoints (0 = disabled)")
    p.add_argument("--no-shuffle", action="store_true", help="disable batch shuffling (debugging only)")
    p.add_argument("--log-file", default="", help="optional CSV file to append step/loss/val_loss/lr/grad_norm")
    p.add_argument("--resume", default="", help="path to an existing .orso checkpoint to continue training from")
    args = p.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")

    resuming = bool(args.resume)
    if resuming:
        state = load_checkpoint(args.resume)
        if state.tokenizer is None:
            raise SystemExit("--resume checkpoint has no tokenizer; cannot re-encode the corpus")
        tokenizer = state.tokenizer
        model = state.model
        optimizer = state.optimizer
        trainer = state.build_trainer()

        if trainer.steps >= args.steps:
            raise SystemExit(
                f"checkpoint is already at step {trainer.steps}, which is >= --steps {args.steps}; "
                "pass a larger --steps to keep training"
            )

        # Extend the existing scheduler to the new target instead of rebuilding
        # it from step 0 (that would replay warmup/cosine from scratch and,
        # combined with the trainer's already-elapsed step count, could leave
        # the optimizer's LR at the wrong point in the curve).
        scheduler = trainer.scheduler
        if scheduler is not None:
            scheduler.total_steps = int(args.steps)
        else:
            scheduler = CosineScheduler(optimizer, total_steps=args.steps, warmup_steps=0)
            trainer.scheduler = scheduler

        print(f"resumed from {args.resume} at step={trainer.steps}, target steps={args.steps}")
    else:
        tokenizer = BPETokenizer(vocab_size=max(256, args.vocab_size), min_frequency=2).train([text])
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

    ids = tokenizer.encode(text)
    train_ids, val_ids = split_token_ids(ids, args.val_fraction)
    context = model.config.context_length
    if len(train_ids) <= context or len(val_ids) <= context:
        raise SystemExit(
            "corpus too small for this --context/--val-fraction: both the train and validation "
            "slices must contain more tokens than --context. Use a bigger corpus, a smaller "
            "--context, or a smaller --val-fraction."
        )
    train_dataset = CausalDataset(train_ids, context)
    val_dataset = CausalDataset(val_ids, context)

    eval_every = args.eval_every or max(1, args.steps // 10)
    eval_batch_size = args.eval_batch_size or args.batch_size

    checkpoint_path = Path(args.checkpoint)
    last_path = _sibling_path(checkpoint_path, "last")
    best_path = _sibling_path(checkpoint_path, "best")
    best_val_loss = float("inf")

    def do_save(path: Path, extra_meta: dict) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            tokenizer=tokenizer,
            trainer=trainer,
            scheduler=scheduler,
            metadata={"phase": 7, "script": "scripts/train.py", "text_chars": len(text), **extra_meta},
        )

    log_handle, log_writer = _open_log(args.log_file, resuming)

    start_step = trainer.steps
    batches = _epoch_batches(train_dataset, args.batch_size, shuffle=not args.no_shuffle, seed=args.seed)

    try:
        while trainer.steps < args.steps:
            inputs, targets = next(batches)
            last = trainer.train_batch(inputs, targets)
            step = int(last["step"])

            val_loss = None
            if step % eval_every == 0 or step == args.steps:
                val_loss = evaluate(trainer, val_dataset, eval_batch_size)
                print(
                    f"step={step} loss={last['loss']:.6f} val_loss={val_loss:.6f} "
                    f"lr={last['lr']:.6g} grad={last['grad_norm']:.6f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    do_save(best_path, {"val_loss": best_val_loss, "kind": "best"})
                    print(f"  new best (val_loss={best_val_loss:.6f}) -> {best_path}")

            if args.save_every and step % args.save_every == 0:
                do_save(last_path, {"kind": "last"})

            if log_writer is not None:
                log_writer.writerow([step, f"{last['loss']:.6f}", "" if val_loss is None else f"{val_loss:.6f}", f"{last['lr']:.6g}", f"{last['grad_norm']:.6f}"])
                log_handle.flush()
    finally:
        if log_handle is not None:
            log_handle.close()

    do_save(checkpoint_path, {"kind": "final"})
    trained_steps = trainer.steps - start_step
    print(f"saved={checkpoint_path} (final, step={trainer.steps})")
    print(f"best={best_path} (val_loss={best_val_loss:.6f})" if best_val_loss != float("inf") else "best=none (no eval ran)")
    print(f"trained {trained_steps} new steps this run")


if __name__ == "__main__":
    main()

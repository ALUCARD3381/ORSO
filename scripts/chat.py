#!/usr/bin/env python3
"""Interactive ORSO terminal chat."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orso.chat import ChatSession
from orso.runtime import ORSORuntime


def main() -> None:
    p = argparse.ArgumentParser(description="ORSO complete CLI chat")
    p.add_argument("checkpoint")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system", default="You are ORSO, a local language model.")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--session", default="")
    args = p.parse_args()

    session = ChatSession(system_prompt=args.system, max_turns=args.max_turns, max_chars=args.max_chars)
    runtime = ORSORuntime.from_checkpoint(args.checkpoint, session=session)
    if args.session and Path(args.session).exists():
        runtime.load_session(args.session)

    st = runtime.stats
    print("╔══════════════════════════════════════╗")
    print("║              O R S O                ║")
    print("║        local mini-LLM runtime       ║")
    print("╚══════════════════════════════════════╝")
    print(f"params={st.parameter_count} vocab={st.vocab_size} context={st.context_length} step={st.trainer_steps}")
    print("/exit  /clear  /save  /stats")

    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/clear":
            runtime.session.clear()
            print("Context cleared.")
            continue
        if line == "/stats":
            st = runtime.stats
            print(f"params={st.parameter_count} vocab={st.vocab_size} context={st.context_length} step={st.trainer_steps}")
            continue
        if line == "/save":
            path = args.session or "session.orso.json"
            runtime.save_session(path)
            print(f"Session saved: {path}")
            continue
        try:
            answer = runtime.respond(
                line,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
            print(f"ORSO> {answer}")
        except Exception as exc:
            print(f"ORSO error: {exc}", file=sys.stderr)
            raise

    if args.session:
        runtime.save_session(args.session)


if __name__ == "__main__":
    main()

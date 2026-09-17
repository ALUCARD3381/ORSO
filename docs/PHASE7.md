# ORSO Phase 7 — Final Runtime

Phase 7 finalizes the runtime with:

- unified checkpoint/tokenizer/model runtime;
- bounded chat session memory;
- temperature, top-k and top-p sampling;
- native incremental KV-cache per Transformer layer;
- automatic cache rebuild when the sliding context window is exceeded;
- `/exit`, `/clear`, `/save` and `/stats` chat commands;
- green/cyan ANSI terminal banner with `--no-color` and `NO_COLOR` fallback;
- deterministic cache-vs-non-cache integration tests.

## KV-cache semantics

`Model.forward()` remains the training/reference path. `Model.forward_cached()` is inference-only and keeps rotated K plus V tensors for every Transformer layer. New tokens calculate only their query and new K/V, then attend against the cached history.

When the active context reaches `context_length`, the runtime rebuilds the cache from the newest `context_length - 1` tokens plus the new token. This keeps the cached path numerically aligned with the existing sliding-window reference path and restarts RoPE positions consistently.

Use `model.reset_kv_cache()` to explicitly clear the cache. `ORSORuntime.clear_session()` and `load_session()` also reset it.

## CLI colors

`scripts/chat.py` uses ANSI green (`92`) and cyan (`96`) for the banner and prompt/output accents. Pass `--no-color` or set `NO_COLOR=1` to disable them.

# ORSO Fase 6 — Checkpoints + Inference

A Fase 6 adiciona persistência reproduzível do estado de treino e uma camada de inferência autoregressiva.

## O que entra

### Checkpoint

O formato `.orso` é um ZIP versionado contendo:

- `metadata.json` — versão do formato, configuração do modelo, hiperparâmetros do AdamW, steps, scheduler e metadados.
- `weights.bin` — todos os parâmetros em `float32`.
- `optimizer.bin` — estados `m` e `v` do AdamW.
- `tokenizer.json` — estado completo do BPE quando fornecido.

O carregamento valida versão, contagem/tamanho dos parâmetros e SHA-256 dos payloads binários.

### Resume

`LoadedCheckpoint.build_trainer()` reconstrói `Trainer`, incluindo optimizer e scheduler restaurados, para continuar o treinamento a partir do step salvo.

### Inferência

`orso.inference` fornece:

- `generate_ids()` para geração por IDs.
- `generate_text()` para prompt + tokenizer.
- greedy quando `temperature=0`.
- amostragem por temperatura e `top_k` quando `temperature>0`.
- janela causal limitada por `context_length`.
- `eos_token_id` opcional.

## Uso básico

```python
from orso.checkpoint import save_checkpoint

save_checkpoint(
    "checkpoints/orsO.orso",
    model,
    optimizer,
    tokenizer=tokenizer,
    trainer=trainer,
    scheduler=trainer.scheduler,
)
```

Para carregar:

```python
from orso.checkpoint import load_checkpoint

state = load_checkpoint("checkpoints/orsO.orso")
trainer = state.build_trainer()
```

Para inferência:

```python
from orso.inference import generate_text

text = generate_text(
    state.model,
    state.tokenizer,
    "hello",
    max_new_tokens=32,
    temperature=0.0,
)
```

O formato foi desenhado para depender apenas da biblioteca padrão Python e do núcleo ORSO já existente.

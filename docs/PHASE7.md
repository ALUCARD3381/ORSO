# ORSO Fase 7 — Complete ORSO

A Fase 7 fecha o fluxo de execução em uma camada única: checkpoint → tokenizer → modelo → memória de sessão → inference → CLI.

## Componentes

- `orso/chat.py`: memória textual limitada por turnos e caracteres.
- `orso/runtime.py`: runtime unificado para checkpoint, sessão, inferência e estatísticas.
- `orso/inference.py`: sampling greedy, temperature, top-k e top-p.
- `scripts/chat.py`: chat interativo de terminal.
- `scripts/train.py`: primeiro fluxo de treino + checkpoint em uma única CLI.
- `scripts/inspect.py`: inspeção de checkpoint sem iniciar chat.
- `tests/test_phase7.py`: validação do pipeline integrado.

## Limite de contexto

A memória de sessão guarda os turnos mais recentes e reduz o prompt ao orçamento configurado. Na geração, o modelo continua respeitando `context_length` e usa a janela final de tokens. Não é afirmado aqui um KV-cache nativo; a implementação usa recomputação causal da Fase 6.

## Chat

```bash
PYTHONPATH="$PWD" python scripts/chat.py checkpoints/model.orso \
  --temperature 0.7 --top-k 40 --top-p 0.95 --tokens 64
```

Com persistência da conversa:

```bash
PYTHONPATH="$PWD" python scripts/chat.py checkpoints/model.orso --session data/session.json
```

## Treino integrado

```bash
PYTHONPATH="$PWD" python scripts/train.py data/corpus.txt checkpoints/model.orso \
  --steps 1000 --context 32
```

## Inspeção

```bash
PYTHONPATH="$PWD" python scripts/inspect.py checkpoints/model.orso
```

## Validação

A Fase 7 só deve virar commit depois de:

1. build C++ no ARMv7;
2. testes das Fases 1–7 passando;
3. chat carregando um checkpoint real;
4. inspeção do checkpoint sem erro;
5. `git status` revisado para excluir ZIPs e artefatos temporários.

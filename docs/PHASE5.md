# ORSO Fase 5 — Modelo + treino

Esta entrega empacota o primeiro modelo causal completo e o caminho de treinamento sem frameworks externos.

## Núcleo C++20

- `TransformerModel`: embedding, múltiplos blocos Transformer, RMSNorm, MHA causal com RoPE, SwiGLU, RMSNorm final e projeção para vocabulário.
- `cross_entropy`: loss causal estável com backward direto para os logits.
- `AdamW`: atualização nativa com momentos, bias correction, weight decay desacoplado e clipping opcional por norma global.

## Python

- `orso.model.ModelConfig`: configuração reproduzível do modelo.
- `orso.tokenizer.BPETokenizer`: BPE byte-level sem dependências externas, com save/load JSON.
- `orso.dataset.CausalDataset`: janelas de contexto para next-token prediction.
- `orso.training.Trainer`: loop de uma batch, métricas e scheduler coseno opcional.
- `scripts/train_toy.py`: treino pequeno para validação funcional.

## Contratos

### Model

```text
input tokens: [B, T]
logits:        [B, T, V]
```

### Cross-Entropy

```text
logits:  [B, T, V]
targets: [B, T]
loss:    [1]
```

### Optimizer

`AdamW.parameters` deve receber tensores com `requires_grad=True`.

## Validação

O pacote deve ser aplicado no checkout atual da Fase 4 e validado no ARMv7 com:

```bash
cd ~/ORSO
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . -j2
cd ..
PYTHONPATH="$PWD" python tests/test_step1.py
PYTHONPATH="$PWD" python tests/test_phase2.py
PYTHONPATH="$PWD" python tests/test_phase3.py
PYTHONPATH="$PWD" python tests/test_phase4.py
PYTHONPATH="$PWD" python tests/test_phase5.py
```

A Fase 5 só deve ser considerada concluída depois de todos os cinco testes passarem no ARMv7.

# ORSO

**Local mini-LLM from scratch for Android/Termux ARMv7**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-informational)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-informational)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-build-informational)](https://cmake.org/)
[![Architecture](https://img.shields.io/badge/target-ARMv7%20%2B%20NEON-informational)](https://developer.arm.com/documentation/)

**Repository:** `ALUCARD3381/ORSO`
**Release:** `0.7.0`

ORSO é um projeto de pesquisa/engenharia para construir uma pequena LLM local, do zero, com uma arquitetura híbrida **C++20 + Python**. O núcleo numérico pesado vive em C++ e é exposto ao Python através de **pybind11**; a camada Python fica responsável pela composição do modelo, tokenizer, dataset, treino, checkpoints, inferência e CLI de chat.

> **Objetivo:** maximizar controle, portabilidade e desempenho em hardware móvel limitado, sem depender de frameworks externos de machine learning como PyTorch, TensorFlow ou JAX.

---

## Estado atual

| Fase | Estado | Conteúdo |
|---|---|---|
| 1 | ✅ Concluída | Fundação C++20/CMake/pybind11 |
| 2 | ✅ Concluída | Tensor Engine float32 |
| 3 | ✅ Concluída | Autograd + kernels NEON |
| 4 | ✅ Concluída | Embeddings + RoPE + RMSNorm + MHA + SwiGLU |
| 5 | ✅ Concluída | Modelo completo + AdamW + BPE + treino |
| 6 | ✅ Concluída | Checkpoints `.orso` + inferência + sampling + KV-cache |
| 7 | ✅ Concluída | CLI/chat + memória de contexto + integração final |

O runtime está completo de ponta a ponta: tensor engine → autograd → Transformer → treino → checkpoint → inferência com KV-cache → chat interativo em terminal.

---

## Arquitetura

```text
ORSO/
├── core/
│   ├── include/orso/
│   │   ├── tensor.hpp
│   │   ├── neon_kernels.hpp
│   │   ├── transformer.hpp
│   │   ├── model.hpp
│   │   └── optimizer.hpp
│   ├── src/
│   │   ├── tensor.cpp
│   │   ├── neon_kernels.cpp
│   │   ├── transformer.cpp
│   │   ├── model.cpp
│   │   └── optimizer.cpp
│   └── CMakeLists.txt
├── bindings/
│   └── bindings.cpp
├── orso/
│   ├── __init__.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── training.py
│   ├── checkpoint.py
│   ├── inference.py
│   ├── runtime.py
│   └── chat.py
├── scripts/
│   ├── train.py
│   ├── train_toy.py
│   ├── infer.py
│   ├── chat.py
│   └── inspect_checkpoint.py
├── tests/
│   ├── test_step1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   └── test_phase7.py
├── data/
├── checkpoints/
├── docs/
│   ├── PHASE4.md
│   ├── PHASE5.md
│   ├── PHASE6.md
│   └── PHASE7.md
├── CMakeLists.txt
└── README.md
```

### Divisão de responsabilidades

**C++20 (`core/`, `bindings/`)**
- Tensor Engine float32 com autograd e construção do grafo
- Matmul/batched matmul
- Kernels SIMD/NEON quando disponíveis
- Softmax, RMSNorm, RoPE, Embedding lookup
- Multi-Head Attention causal
- SwiGLU
- `TransformerModel` completo (embedding + N blocos + RMSNorm final + projeção de vocabulário)
- `cross_entropy` com backward direto
- `AdamW` nativo (momentos, bias correction, weight decay desacoplado, clipping por norma global)
- KV-cache incremental por camada (`forward_cached`)

**Python (`orso/`, `scripts/`)**
- `orso.model.ModelConfig` — configuração reproduzível do modelo
- `orso.tokenizer.BPETokenizer` — BPE byte-level sem dependências externas, save/load JSON
- `orso.dataset.CausalDataset` — janelas de contexto para next-token prediction
- `orso.training.Trainer` + `CosineScheduler` — loop de treino com scheduler e métricas
- `orso.checkpoint` — formato `.orso` (ZIP versionado, SHA-256 nos binários)
- `orso.inference` — geração por IDs/texto, greedy, temperatura, top-k, top-p
- `orso.runtime.ORSORuntime` — runtime unificado (checkpoint + tokenizer + modelo + sessão)
- `orso.chat.ChatSession` — memória de conversa limitada por turnos/caracteres
- `scripts/chat.py` — CLI interativo com banner ANSI verde/ciano

---

## Modelo-alvo

Hiperparâmetros de referência para a primeira arquitetura completa (ajustáveis via `ModelConfig` ou flags de `scripts/train.py`):

```text
d_model         = 64
num_heads       = 8
hidden_dim      = 256
num_layers      = 6
context_length  = 32
vocab_size      = 768
```

`scripts/train.py` usa por padrão uma configuração menor (`d_model=32`, `heads=4`, `hidden=64`, `layers=2`, `context=16`), adequada para validação rápida em hardware limitado; os valores acima são o alvo para o primeiro modelo "de verdade".

---

## Build no Termux / Ubuntu / ARMv7

```bash
cd ~/ORSO
source venv/bin/activate
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . -j2
```

O resultado esperado é uma extensão Python semelhante a:

```text
orso_core.cpython-3xx-arm-linux-gnueabihf.so
```

Depois:

```bash
cd ~/ORSO
PYTHONPATH="$PWD" python -c "import orso_core; print(orso_core.hello())"
```

Saída esperada:

```text
ORSO core OK
```

---

## Testes

Execute todos os testes pela raiz:

```bash
cd ~/ORSO
for t in tests/test_step1.py tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py \
         tests/test_phase5.py tests/test_phase6.py tests/test_phase7.py; do
    PYTHONPATH="$PWD" python "$t" || exit 1
done
```

- **Fase 4**: embedding + gradient, invariância de norma do RoPE, RMSNorm + backward, SwiGLU + backward, MHA causal + RoPE + backward, disponibilidade do NEON.
- **Fase 5**: `TransformerModel` completo, `cross_entropy`, `AdamW`, `BPETokenizer`, `CausalDataset`, `Trainer`.
- **Fase 6**: round-trip de checkpoint `.orso` (pesos, optimizer, tokenizer, metadados, verificação SHA-256), resume de treino, `generate_ids`/`generate_text` (greedy/temperatura/top-k).
- **Fase 7**: runtime unificado, KV-cache (equivalência numérica entre `forward` e `forward_cached`), rebuild automático de cache ao ultrapassar a janela de contexto, sessão de chat com persistência.

---

## Uso rápido

Treinar um checkpoint a partir de um corpus de texto:

```bash
PYTHONPATH="$PWD" python scripts/train.py corpus.txt checkpoints/modelo.orso \
    --vocab-size 512 --context 16 --d-model 32 --heads 4 --hidden 64 --layers 2 \
    --steps 200 --batch-size 4 --lr 3e-4
```

Gerar texto a partir de um checkpoint:

```bash
PYTHONPATH="$PWD" python scripts/infer.py checkpoints/modelo.orso "Era uma vez" \
    --tokens 64 --temperature 0.8 --top-k 40
```

Chat interativo com KV-cache e memória de sessão:

```bash
PYTHONPATH="$PWD" python scripts/chat.py checkpoints/modelo.orso \
    --temperature 0.8 --top-k 40 --top-p 0.95 --tokens 64 --session sessao.json
```

Comandos disponíveis no chat: `/exit`, `/clear`, `/save`, `/stats`. Use `--no-color` ou `NO_COLOR=1` para desativar as cores ANSI.

---

## API mínima

Exemplo de uso direto em Python (núcleo C++ via pybind11):

```python
import orso_core

Tensor = orso_core.Tensor

x = Tensor.random_normal([2, 4, 64], seed=123, requires_grad=True)
w = Tensor.ones([64], requires_grad=True)

x_norm = orso_core.rmsnorm(x, w)
loss = x_norm.sum()
loss.backward()
```

Attention:

```python
out = orso_core.multi_head_attention(
    x, wq, wk, wv, wo,
    num_heads=8,
    causal=True,
)
```

SwiGLU:

```python
out = orso_core.swiglu(x, w_gate, w_up, w_down)
```

Camada de alto nível (Python), do checkpoint ao chat:

```python
from orso.runtime import ORSORuntime
from orso.chat import ChatSession

session = ChatSession(system_prompt="Você é o ORSO.", max_turns=12)
runtime = ORSORuntime.from_checkpoint("checkpoints/modelo.orso", session=session)

resposta = runtime.respond("Olá!", max_new_tokens=64, temperature=0.8, top_k=40)
print(resposta)
```

---

## Checkpoints (`.orso`)

O formato `.orso` é um ZIP versionado contendo:

- `metadata.json` — versão do formato, configuração do modelo, hiperparâmetros do AdamW, steps, scheduler e metadados.
- `weights.bin` — todos os parâmetros em `float32`.
- `optimizer.bin` — estados `m` e `v` do AdamW.
- `tokenizer.json` — estado completo do BPE quando fornecido.

O carregamento valida versão, contagem/tamanho dos parâmetros e SHA-256 dos payloads binários. `LoadedCheckpoint.build_trainer()` reconstrói o `Trainer` (optimizer + scheduler) para retomar o treino a partir do step salvo.

---

## KV-cache e inferência

`Model.forward()` continua sendo o caminho de treino/referência. `Model.forward_cached()` é exclusivo para inferência e mantém K rotacionado + V em cache por camada do Transformer. Tokens novos calculam apenas sua própria query e K/V, e atendem contra o histórico em cache.

Quando o contexto ativo atinge `context_length`, o runtime reconstrói o cache a partir dos últimos `context_length - 1` tokens mais o novo token, mantendo o caminho com cache numericamente alinhado ao caminho de janela deslizante de referência e reiniciando as posições do RoPE de forma consistente.

Use `model.reset_kv_cache()` para limpar o cache explicitamente. `ORSORuntime.clear_session()` e `load_session()` também o reiniciam.

---

## Performance e NEON

O build detecta suporte a NEON durante o `cmake` e habilita os kernels SIMD quando o compilador/arquitetura disponibilizam `<arm_neon.h>`. O NEON é aplicado às operações elementwise críticas, ao dot product e ao matmul 2D/batched, com fallback escalar para ambientes sem NEON.

Para verificar no Python:

```bash
PYTHONPATH="$PWD" python -c "import orso_core; print('NEON:', orso_core.neon_enabled())"
```

---

## Princípios de engenharia

1. **Sem framework de ML**: o cálculo é implementado pelo próprio projeto.
2. **API estável entre fases**: as extensões são adicionadas sem abandonar a interface do Tensor Engine.
3. **Autograd explícito**: cada operação diferenciável registra seus pais e a regra de backward.
4. **ARM-first**: o código considera ARMv7/NEON como alvo principal, mas mantém fallback escalar para desenvolvimento e validação.
5. **Testes por componente**: cada fase é validada a partir do Python antes de seguir para a próxima.
6. **Git desde o início**: cada fase é entregue como um pacote fechado e vira um commit independente.

---

## Desenvolvimento no telemóvel

Para sessões longas de compilação/treino em Termux, o fluxo recomendado é manter o bloqueio de suspensão ativo:

```bash
termux-wake-lock
```

Ao terminar a sessão:

```bash
termux-wake-unlock
```

---

## Repositório e contribuição

O desenvolvimento seguiu um histórico por fases (1 a 7), cada uma consolidada em um commit próprio, com testes reproduzíveis no alvo ARMv7.

Para reportar um problema, inclua: arquitetura, versão do Python, versão do GCC, versão do CMake, saída completa do `cmake`, saída do teste que falhou e se `orso_core.neon_enabled()` retornou `True` ou `False`.

## Licença

MIT — consulte `LICENSE`.

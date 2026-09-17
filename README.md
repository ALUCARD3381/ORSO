# ORSO

**Local mini-LLM from scratch for Android/Termux ARMv7**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-informational)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-informational)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-build-informational)](https://cmake.org/)
[![Architecture](https://img.shields.io/badge/target-ARMv7%20%2B%20NEON-informational)](https://developer.arm.com/documentation/)

**Repository:** `ALUCARD3381/ORSO`  
**Release:** `0.7.0`

ORSO é um projeto de pesquisa/engenharia para construir uma pequena LLM local, do zero, com uma arquitetura híbrida **C++20 + Python**. O núcleo numérico pesado vive em C++ e é exposto ao Python através de **pybind11**; a camada Python fica responsável pela composição do modelo, tokenizer, dataset, treino e CLI.

> **Objetivo:** maximizar controle, portabilidade e desempenho em hardware móvel limitado, sem depender de frameworks externos de machine learning como PyTorch, TensorFlow ou JAX.

---

## Estado atual

| Fase | Estado | Conteúdo |
|---|---|---|
| 1 | ✅ Concluída | Fundação C++20/CMake/pybind11 |
| 2 | ✅ Concluída | Tensor Engine float32 |
| 3 | ✅ Concluída | Autograd + kernels NEON |
| **4** | **✅ Concluída nesta entrega** | **Embeddings + RoPE + RMSNorm + MHA + SwiGLU** |
| 5 | ⏳ Próxima | Modelo completo + AdamW + BPE + treino |
| 6 | ⏳ | Checkpoints `.orso` + inferência + sampling + KV-cache |
| 7 | ⏳ | CLI/chat + memória de contexto + integração final |

A implementação desta entrega mantém as operações fundamentais das fases anteriores e adiciona os blocos necessários para montar um Transformer pequeno em cima do mesmo núcleo autograd.

---

## Arquitetura

```text
ORSO/
├── core/
│   ├── include/orso/
│   │   ├── tensor.hpp
│   │   ├── neon_kernels.hpp
│   │   └── transformer.hpp
│   ├── src/
│   │   ├── tensor.cpp
│   │   ├── neon_kernels.cpp
│   │   └── transformer.cpp
│   └── CMakeLists.txt
├── bindings/
│   └── bindings.cpp
├── orso/
│   └── __init__.py
├── tests/
│   ├── test_step1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
├── scripts/
├── data/
├── checkpoints/
├── docs/
├── CMakeLists.txt
└── README.md
```

### Divisão de responsabilidades

**C++20**
- Tensor Engine float32
- Autograd e construção do grafo
- Matmul/batched matmul
- Kernels SIMD/NEON quando disponíveis
- Softmax
- RMSNorm
- RoPE
- Embedding lookup
- Multi-Head Attention causal
- SwiGLU

**Python**
- API de alto nível
- Model definition
- Tokenizer BPE
- Dataset
- Loop de treino
- Checkpoints
- CLI/chat

---

## Modelo-alvo

Os hiperparâmetros planejados para a primeira arquitetura completa são:

```text
d_model         = 64
num_heads       = 8
hidden_dim      = 256
num_layers      = 6
context_length  = 32
vocab_size      = 768
```

Esta fase implementa os blocos fundamentais; o empacotamento desses blocos num Transformer completo fica para a Fase 5.

---

## O que entrou na Fase 4

### 1. Embeddings

`embedding(weight, token_ids)` faz lookup de linhas de uma matriz de embeddings e cria o caminho de gradiente para os pesos utilizados.

### 2. RoPE

`rope(x, theta=10000.0)` aplica Rotary Positional Embeddings ao último eixo par do tensor. A operação possui backward analítico.

### 3. RMSNorm

`rmsnorm(x, weight, eps=1e-5)` normaliza pelo RMS do último eixo e suporta backward para entrada e escala aprendida.

### 4. Multi-Head Attention

`multi_head_attention(...)` implementa:

```text
X
 ├─ Wq → Q ─┐
 ├─ Wk → K ─┼→ RoPE → scaled dot-product attention → output projection
 └─ Wv → V ┘
```

Inclui:
- split de heads
- transposição para `[B, H, T, Dh]`
- RoPE em Q/K
- escala `1/sqrt(Dh)`
- máscara causal opcional
- softmax estável
- agregação `P @ V`
- projeção final
- autograd por composição das operações diferenciáveis

### 5. SwiGLU

`swiglu(x, gate_weight, up_weight, down_weight)` implementa o caminho feed-forward:

```text
gate = X @ W_gate
up   = X @ W_up
h    = SiLU(gate) * up
out  = h @ W_down
```

---

## Build no Termux / Ubuntu / ARMv7

O projeto foi desenhado para o fluxo utilizado no desenvolvimento do ORSO:

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
PYTHONPATH="$PWD" python tests/test_step1.py
PYTHONPATH="$PWD" python tests/test_phase2.py
PYTHONPATH="$PWD" python tests/test_phase3.py
PYTHONPATH="$PWD" python tests/test_phase4.py
```

Ou em sequência:

```bash
for t in tests/test_step1.py tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py; do
    PYTHONPATH="$PWD" python "$t" || exit 1
done
```

O teste da Fase 4 verifica:
- embedding lookup + gradient
- invariância de norma por par no RoPE
- normalização RMS
- backward do RMSNorm
- SwiGLU + backward
- MHA causal + RoPE + backward
- estado de disponibilidade do NEON

---

## API mínima

Exemplo de uso direto em Python:

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
    x,
    wq,
    wk,
    wv,
    wo,
    num_heads=8,
    causal=True,
)
```

SwiGLU:

```python
out = orso_core.swiglu(x, w_gate, w_up, w_down)
```

---

## Performance e NEON

O build detecta suporte a NEON durante o `cmake` e habilita os kernels SIMD quando o compilador/arquitetura disponibilizam `<arm_neon.h>`.

Nesta linha de desenvolvimento, o NEON é aplicado diretamente às operações elementwise críticas, ao dot product e ao matmul 2D/batched. A implementação mantém fallback escalar para ambientes sem NEON e deixa espaço para microkernels ARM ainda mais especializados nas próximas otimizações.

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
5. **Testes por componente**: cada fase deve ser validada a partir do Python antes de seguir para a próxima.
6. **Git desde o início**: cada fase é entregue como um pacote fechado e deve virar um commit independente.

---

## Desenvolvimento no telemóvel

Para sessões longas de compilação/treino em Termux, o fluxo recomendado no ambiente do projeto é manter o bloqueio de suspensão ativo:

```bash
termux-wake-lock
```

Ao terminar a sessão:

```bash
termux-wake-unlock
```

---

## Roadmap técnico

### Fase 5 — Modelo + treino

- Transformer completo
- Embedding + posições
- múltiplos blocos
- AdamW nativo
- BPE vocab 768
- dataset próprio
- loop de treino em Python
- scheduler e métricas

### Fase 6 — Checkpoint + inferência

- formato binário `.orso`
- pesos + arquitetura + step
- estado AdamW
- carregamento/reconstrução
- geração autoregressiva
- temperatura
- top-k
- top-p
- KV-cache

### Fase 7 — ORSO completo

- `scripts/chat.py`
- memória de sessão
- banner ASCII
- terminal verde/ciano
- integração end-to-end
- otimizações finais
- primeiro treino completo

---

## Repositório e contribuição

O desenvolvimento segue um histórico por fases. Cada fase deve ser consolidada em um commit próprio antes do próximo ciclo de implementação, mantendo testes reproduzíveis no alvo ARMv7.

Para reportar um problema, inclua: arquitetura, versão do Python, versão do GCC, versão do CMake, saída completa do `cmake`, saída do teste que falhou e se `orso_core.neon_enabled()` retornou `True` ou `False`.

## Licença

MIT — consulte `LICENSE`.

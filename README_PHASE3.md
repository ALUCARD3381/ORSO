# ORSO — Fase 3

Fase 3 implementa, de uma vez, **Autograd em C++** e **kernels SIMD/NEON** mantendo a API pública do Tensor Engine da Fase 2.

## O que foi implementado

### Autograd

- `requires_grad()` e `set_requires_grad()`.
- Gradiente acumulado por tensor-leaf.
- `grad()` e `zero_grad()`.
- `backward()` para saídas escalares.
- `backward(grad)` para saídas não escalares.
- Grafo dinâmico baseado em nós e referências aos pais.
- Topological traversal iterativo via DFS para propagar gradientes.
- Execução interna de `backward()` em modo sem criação de novos grafos (`no-grad`).
- Acumulação correta quando um tensor participa de múltiplos caminhos.

### Gradientes suportados

- soma
- subtração
- multiplicação elementwise
- divisão elementwise
- multiplicação por escalar
- divisão por escalar
- negação
- `reshape`
- `transpose`
- `matmul`: vetor×vetor, matriz×vetor, vetor×matriz e batched N-D

A implementação não adiciona broadcasting nesta fase, preservando a regra de igualdade de shapes do Tensor Engine da Fase 2.

### NEON

`core/src/neon_kernels.cpp` contém kernels com fallback escalar:

- soma elementwise
- subtração elementwise
- multiplicação elementwise
- dot product
- multiplicação matriz-matriz 2D

No ARMv7/32-bit ARM, o CMake tenta habilitar `-mfpu=neon` automaticamente. Em plataformas onde NEON não é aplicável, o mesmo código compila usando o caminho escalar.

## Build

No Termux/Ubuntu, com o `venv` da Fase 1/2 ativado:

```bash
cd ~/ORSO
source venv/bin/activate
rm -rf build
mkdir build
cd build

cmake .. \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build . -j2
```

## Testes

```bash
PYTHONPATH="$PWD" python ../tests/test_step1.py
PYTHONPATH="$PWD" python ../tests/test_phase2.py
PYTHONPATH="$PWD" python ../tests/test_phase3.py
```

Saída esperada do novo teste:

```text
PASS: ORSO Fase 3 — Autograd + NEON completo
```

O método `orso_core.neon_available()` mostra se o módulo atualmente carregado foi compilado com os kernels NEON ativos.

## Limites desta fase

Ainda não fazem parte da Fase 3: AdamW, Transformer, attention, embeddings, RoPE, tokenizer, dataset, checkpoint `.orso` ou loop de treino. Esses componentes ficam nas fases seguintes.

# ORSO — Fase 2: Tensor Engine

Esta fase implementa o Tensor Engine nativo em C++20 e expõe a API ao Python por `pybind11`.

## Incluído

- Tensor N-dimensional em `float32`.
- Armazenamento contíguo row-major.
- Criação por shape/dados, `zeros`, `ones` e `full`.
- `shape`, `ndim`, `size`, `empty`, `data`, `item`, `get`, `set`.
- `reshape` preserva o número de elementos e retorna um tensor contíguo com o novo shape.
- `transpose()` e `transpose(axes)` para N dimensões.
- Operações elementwise: soma, subtração, multiplicação e divisão.
- Operações com escalar: multiplicação e divisão.
- Negação.
- `matmul` para vector/vector, matriz/vector, vector/matriz e batched matmul N-D com prefixos de batch idênticos.
- `fill` e representação textual básica.
- Bindings Python e testes automatizados.

## Fora desta fase

Autograd, kernels NEON, attention, RMSNorm, SwiGLU, RoPE, AdamW, tokenizer e treino ficam para as fases seguintes.

## Compilar no Ubuntu/Termux

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
PYTHONPATH="$PWD" python -c "import orso_core; print(orso_core.hello())"
PYTHONPATH="$PWD" python ../tests/test_step1.py
PYTHONPATH="$PWD" python ../tests/test_phase2.py
```

Resultado esperado da Fase 2:

```text
PASS: ORSO Fase 2 — Tensor Engine completo
```

## Exemplo Python

```python
import orso_core

Tensor = orso_core.Tensor

a = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
b = Tensor([3, 2], [7, 8, 9, 10, 11, 12])

c = a @ b
print(c.shape)   # [2, 2]
print(c.data())  # [58, 64, 139, 154]
```

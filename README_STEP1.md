# ORSO — Passo 1: pybind11

Objetivo: validar que C++20 + pybind11 + Python funcionam no ambiente Ubuntu/Termux.

## 1. Entrar no projeto

```bash
cd ~/ORSO

git init
git add .
git commit -m "chore: inicializa ORSO e teste pybind11"
```

## 2. Criar o ambiente Python

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install pybind11
```

## 3. Configurar o CMake

```bash
rm -rf build
mkdir build
cd build

cmake .. \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

## 4. Compilar

```bash
cmake --build . -j2
```

## 5. Testar pelo Python

Ainda dentro de `build/`:

```bash
PYTHONPATH="$PWD" python -c "import orso_core; print(orso_core.hello())"
```

Resultado esperado:

```text
ORSO core OK
```

## 6. Teste automatizado

```bash
PYTHONPATH="$PWD" python ../tests/test_step1.py
```

Resultado esperado:

```text
PASS: orso_core import/teste OK
```

## 7. Teste adicional

```bash
file orso_core*.so
python -c "import orso_core; print(orso_core.__doc__)"
```

O importante nesta etapa é o `import orso_core` funcionar sem erro.

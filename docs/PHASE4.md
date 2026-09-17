# ORSO Fase 4 — Transformer Core

Esta entrega consolida os blocos do núcleo Transformer previstos para a quarta fase do ORSO.

## Componentes

- Embeddings: lookup diferenciável sobre a matriz de pesos.
- RoPE: rotação 2D por pares no eixo de features, com backward analítico.
- RMSNorm: normalização RMS do último eixo com gradientes para entrada e escala.
- Softmax: implementação estável por máximo local e backward vetorial.
- MHA: projeções Q/K/V, split heads, RoPE, causal mask, scaled dot-product, softmax e output projection.
- SwiGLU: SiLU(gate) * up e projeção de saída.

## Contrato de shapes

### MHA

```text
x:  [B, T, D]
wq: [D, D]
wk: [D, D]
wv: [D, D]
wo: [D, D]

out: [B, T, D]
```

`D % H == 0` é obrigatório.

### RMSNorm

```text
x:      [..., D]
weight: [D]
out:    [..., D]
```

### RoPE

```text
x: [..., T, D]
D: par
out: mesma shape
```

## Nota de performance

A Fase 4 prioriza correção, shape handling e integração com autograd. A próxima etapa pode substituir o matmul genérico por microkernels ARM/NEON especializados para `M x K @ K x N`, attention e projeções do Transformer.

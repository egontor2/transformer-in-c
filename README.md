# Transformer en C

Implementación inicial de un bloque **Transformer encoder** en C99, sin
dependencias externas. El modelo incluye:

- Layer normalization pre-norm.
- Multi-head self-attention escalada.
- Proyección QKV y proyección de salida.
- Feed-forward network con activación ReLU.
- Múltiples capas encoder y conexiones residuales.
- Entrada visual mediante patch embedding configurable.
- Codificación posicional sinusoidal para los patches.
- Cabeza de clasificación y entrenamiento SGD con entropía cruzada.
- Loop de entrenamiento para datasets en memoria.

Los tensores de entrada y salida usan layout row-major:
`[sequence_length][d_model]`.

## Compilar y ejecutar

```sh
make
make run
make test
```

La API pública está en `include/transformer.h`. Para visión, configura
`patch_size` e `input_channels` y llama a `transformer_forward_image`. La
imagen debe estar en layout `[height][width][channels]`; se divide en patches
no solapados y cada patch se proyecta a `d_model`.

La cabeza de clasificación se activa con `n_classes`. El entrenamiento se
realiza con `transformer_train_image`, que actualiza la cabeza mediante SGD y
entropía cruzada sobre el promedio de los tokens codificados. En esta primera
iteración el encoder se usa como backbone congelado; el siguiente paso será
añadir backpropagation completa para ajustar también sus pesos.

Para varias imágenes puede usarse `transformer_train_dataset`, con un buffer
contiguo `[sample][height][width][channels]` y un vector de etiquetas.

También se incluye `include/dataset.h`, un cargador de manifests CSV con
imágenes PGM (`P2` o `P5`):

```text
data/cat_001.pgm,0
data/dog_001.pgm,1
```

`vision_dataset_load_pgm_csv` devuelve los píxeles normalizados a `[0, 1]`,
listos para pasarlos a `transformer_train_dataset`.

## Fase 1: núcleo de tensores

La base de entrenamiento desde cero está organizada en:

- `include/tensor.h`: `Tensor` contiguo con `data`, `grad`, shape y strides;
  además de `Parameter` con buffers para AdamW.
- `include/arena.h`: bump allocator para activaciones temporales, con
  `arena_reset` entre iteraciones.
- `include/ops.h`: GEMM, GEMM con la izquierda transpuesta, residuales y sus
  pases backward acumulativos, LayerNorm, GELU, softmax + cross-entropy y
  AdamW. También incluye `ops_causal_attention`, con máscara causal y
  backward explícito para Q, K y V.

Las operaciones devuelven `0` si tienen shapes compatibles y `-1` ante una
forma inválida. El siguiente bloque de la guía es completar las primitivas
backward del encoder; AdamW ya está disponible mediante
`ops_adamw_step`.

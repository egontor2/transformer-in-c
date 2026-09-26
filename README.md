# Transformer en C

Implementación inicial de un bloque **Transformer encoder** en C99, sin
dependencias externas. El modelo incluye:

- Layer normalization pre-norm.
- Multi-head self-attention escalada.
- Proyección QKV y proyección de salida.
- Feed-forward network con activación ReLU.
- Múltiples capas encoder y conexiones residuales.
- Entrada visual mediante patch embedding configurable.
- API ViT con token `[CLS]`, posiciones aprendibles, bloque Pre-LN y cabeza
  lineal de clasificación.
- Cache de activaciones y backward completo del bloque ViT, incluyendo biases
  affine, atención bidireccional y MLP GELU.
- `ViTModel` entrenable end-to-end: patch projection, embeddings, bloques,
  LayerNorm final y clasificación sobre `[CLS]`.
- `vit_model_train_batch` conecta cross-entropy, backward y AdamW sobre todo
  el backbone y la cabeza.
- `vit_model_save` y `vit_model_load` guardan configuración, pesos, momentos
  AdamW y contador de pasos en checkpoints binarios versionados.
- `vit_model_clip_gradients` aplica clipping global por norma antes de cada
  actualización AdamW del entrenamiento end-to-end.
- `vit_model_evaluate` calcula accuracy, cross-entropy media y throughput de
  inferencia mediante un benchmark reproducible sobre un batch.
- Backend opcional MPS para GEMM en macOS/Apple Silicon, aislado mediante ABI C
  y Objective-C++.
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

### Probar con OrganSMNIST

OrganSMNIST de MedMNIST se distribuye como un `.npz` con las claves
`train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` y
`test_labels`. El repositorio incluye un conversor que evita añadir un parser
NPZ al runtime C:

```sh
python3 -m pip install numpy
python3 scripts/organ_smnist_to_pgm.py organmnist.npz data/organ_smnist
```

Esto genera `train.csv`, `val.csv`, `test.csv` y ficheros PGM grayscale de
28x28. El modelo ViT debe configurarse con `channels=1`, `height=28`,
`width=28`, `patch_size` divisor de 28 (por ejemplo 4), y `classes=11`.
Después, carga el split con `vision_dataset_load_pgm_csv`; sus imágenes están
aplanadas por muestra en el orden `[channel][height][width]`, que coincide con
la entrada NCHW de `vit_model_train_batch` y `vit_model_evaluate`.
El entrenamiento se realiza por batches contiguos: crea un
`ViTModelCache` del tamaño del batch, llama a `vit_model_train_batch` para cada
segmento de `images` y `labels`, y usa `vit_model_evaluate` sobre validación o
test. Para el primer experimento conviene usar `d_model=64`, `heads=4`,
`layers=2` y `patch_size=4`; el modelo resultante tiene 49 tokens por imagen.

El flujo completo está disponible en `examples/organ_smnist.c`:

```sh
make organ-smnist
./build/organ_smnist data/organ_smnist/train.csv \
  data/organ_smnist/val.csv data/organ_smnist/test.csv \
  build/organ_smnist.vit
```

El ejemplo mezcla el dataset con una semilla fija al principio de cada época.
Solo `train.csv` actualiza los pesos, `val.csv` selecciona y conserva el mejor
checkpoint, y `test.csv` se consulta una sola vez al final. Los valores por defecto son
batches completos de 32 muestras, cinco épocas, `learning_rate=0.0001` y
`weight_decay=0.01`. Se pueden ajustar:

```sh
./build/organ_smnist TRAIN.csv VAL.csv TEST.csv CHECKPOINT \
  BATCH EPOCHS LR DECAY WARMUP_EPOCHS SCHEDULER PATIENCE FACTOR [RESUME] [EARLY_STOPPING]
```

El entrenamiento usa por defecto una época de warmup lineal y
`ReduceLROnPlateau`: reduce el learning rate cuando la pérdida de validación no
mejora. `SCHEDULER` puede ser `plateau`, `cosine` o `constant`; los valores por
defecto son `PATIENCE=2` y `FACTOR=0.5`. Para desactivar el warmup usa `0`.

Para reanudar desde un checkpoint existente, añade su ruta como último
argumento. Se restauran los pesos, estados `m/v` y contadores de AdamW:

```sh
./build/organ_smnist_mps TRAIN.csv VAL.csv TEST.csv \
  build/organ_smnist.mps.vit 32 10 0.0001 0.01 1 plateau 2 0.5 \
  build/organ_smnist.mps.vit
```

El entrenamiento guarda además `CHECKPOINT.state` con la época del mejor
checkpoint, el learning rate efectivo, la mejor `val_loss` y el estado de
`ReduceLROnPlateau`. Al usar `--resume`, ese fichero se restaura junto con los
pesos; si falta, se mantienen los valores proporcionados por CLI.

Al reanudar no se repite el warmup. Para forzar deliberadamente otro learning
rate inicial usa `--resume-lr`, por ejemplo `--resume-lr 0.00001`.

También se puede usar la forma nombrada, recomendada para no depender del
orden de los argumentos:

```sh
./build/organ_smnist_mps \
  --train data/organ_smnist/train.csv \
  --val data/organ_smnist/val.csv \
  --test data/organ_smnist/test.csv \
  --checkpoint build/organ_smnist.mps.continued.vit \
  --batch 32 --epochs 20 --lr 0.0001 --weight-decay 0.01 \
  --warmup 1 --scheduler plateau --patience 2 --factor 0.5 \
  --resume build/organ_smnist.mps.vit --early-stopping 6
```

`--early-stopping N` detiene el entrenamiento después de `N` épocas sin
mejora de `val_loss`; `0` lo desactiva.

Al finalizar también se imprime la accuracy por clase del split de test, lo
que permite detectar clases que el accuracy global oculta.

`--metrics-csv PATH` guarda las métricas de cada época (`learning_rate`,
`train_loss`, `val_accuracy` y `val_loss`) para comparar ejecuciones y
graficar la convergencia.

Para compensar desbalance de clases puede usarse
`--class-weights balanced`; calcula pesos inversamente proporcionales a la
frecuencia del split de entrenamiento.

No se activa aumentación geométrica automáticamente: OrganSMNIST incluye clases
laterales (`left`/`right`) y un flip horizontal puede cambiar la etiqueta
anatómica. Una futura aumentación debe ser consciente de la lateralidad y
validarse por clase antes de usarse en entrenamiento.

Si el número de muestras no es múltiplo del batch, descarta únicamente el
último batch incompleto; esto mantiene el cache de activaciones con tamaño
fijo. La mezcla es importante porque los manifests generados por carpetas
agrupan inicialmente todas las imágenes de una clase.
El checkpoint se selecciona por la menor `val_loss` de `val.csv`, alineado con
`ReduceLROnPlateau` y early stopping; después se restaura ese checkpoint y se
informa `test.csv` una sola vez como métrica final.

Si el ZIP de Kaggle contiene imágenes en carpetas `train/0`, `train/1`, etc.,
usa el conversor alternativo:

```sh
python3 -m pip install pillow
python3 scripts/images_to_pgm.py data/organsmnist_raw data/organ_smnist
```

La misma estructura debe existir para `val` y `test`. Si Kaggle solo incluye
`train` y `test`, ejecuta `--splits train test` y reserva una parte del train
para validación antes de entrenar. Las imágenes se convierten a grayscale
28x28 y las etiquetas se asignan según el orden numérico de las carpetas.

## Fase 1: núcleo de tensores

La base de entrenamiento desde cero está organizada en:

- `include/tensor.h`: `Tensor` contiguo con `data`, `grad`, shape y strides;
  además de `Parameter` con buffers para AdamW.
- `include/arena.h`: bump allocator para activaciones temporales, con
  `arena_reset` entre iteraciones.
- `include/ops.h`: GEMM, GEMM con la izquierda transpuesta, residuales y sus
  pases backward acumulativos, LayerNorm, GELU, softmax + cross-entropy y
  AdamW, además de suma de bias y backward. También incluye atención causal y
  atención multi-head bidireccional, con backward explícito para Q, K y V.

Las operaciones devuelven `0` si tienen shapes compatibles y `-1` ante una
forma inválida. `ViTEncoderBlockCache` conserva las activaciones necesarias
para `vit_encoder_block_backward`; los gradientes se acumulan y deben
reinicializarse con `tensor_zero_grad` o `parameter_zero_grad` antes de cada
iteración. AdamW ya está disponible mediante `ops_adamw_step`.

La ruta nueva se configura con `ViTConfig` y se ejecuta con
`vit_model_cache_init`, `vit_model_forward` y `vit_model_backward`. Las
imágenes usan layout NCHW `[batch][channel][height][width]`; la pérdida puede
calcularse sobre `cache.logits` con `ops_softmax_cross_entropy` antes del
backward. `ViTModel` mantiene la API legacy separada para no mezclar su
promedio de tokens ni su encoder congelado con el camino ViT.
El paso de entrenamiento usa weight decay únicamente en matrices de pesos;
biases, parámetros de LayerNorm, `[CLS]` y posiciones quedan sin decay.
Los checkpoints requieren cargar sobre un `ViTModel` con la misma configuración;
la API rechaza archivos incompatibles o truncados.
La inicialización del modelo es determinista y la evaluación no modifica
parámetros ni estados del optimizador.

## Aceleración MPS en macOS

El backend opcional `mps_backend` usa `MPSMatrixMultiplication` sobre buffers
Metal compartidos. El núcleo C y la ruta CPU no dependen de frameworks Apple:

```sh
make mps-test
```

El target compila `src/mps_backend.mm`, enlaza Foundation, Metal y
MetalPerformanceShaders, y ejecuta una comprobación numérica. Si no existe
GPU Metal, `mps_backend_create` devuelve disponibilidad no soportada en lugar
de producir resultados parciales.

Esta integración expone GEMM explícitamente; el binario CPU continúa disponible
como referencia y el binario MPS activa el dispatch durante el entrenamiento.
El backend reutiliza buffers compartidos por clave `(filas, columnas
interiores, columnas resultado)`, con un límite de 16 formas para evitar un
crecimiento ilimitado. Los datos siguen copiándose en cada llamada y la
operación espera síncronamente a la GPU.

Para entrenar con el dispatch MPS activo usa el binario específico de macOS:

```sh
make build/organ_smnist_mps
caffeinate -dismu ./build/organ_smnist_mps \
  data/organ_smnist/train.csv \
  data/organ_smnist/val.csv \
  data/organ_smnist/test.csv \
  build/organ_smnist.mps.vit \
  32 5 0.0001 0.01
```

Este binario instala MPS antes del primer `vit_model_train_batch` y lo mantiene
activo durante entrenamiento y evaluación. Las multiplicaciones GEMM forward y
las dos multiplicaciones del backward (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`) pasan por
`MPSMatrixMultiplication`; operaciones no GEMM permanecen en C como fallback.
La ruta sigue siendo síncrona y copia operandos por operación, por lo que es una
aceleración funcional y no todavía una residencia completa del modelo en GPU.

La arquitectura objetivo sigue el patrón habitual de MPS:

1. Crear `MTLBuffer` persistentes para pesos, activaciones, gradientes y estados
   del optimizador.
2. Construir `MPSMatrix` sobre esos buffers y reutilizar sus descriptores.
3. Encadenar operaciones en un mismo `MTLCommandBuffer` sin sincronizar con la
   CPU entre cada GEMM.
4. Usar `MPSGraph` para el bloque ViT completo cuando las formas sean estáticas,
   incluyendo atención, LayerNorm, activaciones y backward. Esto todavía no
   está implementado en este backend.
5. Descargar únicamente las métricas, checkpoints o resultados solicitados.

La migración requiere añadir referencias de dispositivo a los tensores y una
política explícita de sincronización CPU/GPU; no es correcto simularla copiando
cada `Tensor` a un buffer temporal.

El backend actual reutiliza buffers Metal por forma GEMM, reduciendo
allocaciones repetidas. Todavía copia los datos CPU↔Metal y espera cada
command buffer; la residencia completa de activaciones y la ejecución
asíncrona siguen siendo trabajo futuro.

La API `ops_set_gemm_backend` permite probar el dispatch MPS en operaciones GEMM
del núcleo sin enlazar Metal en la librería CPU. Es un backend global y debe
instalarse solo alrededor de trabajo de un único hilo; `ops_reset_gemm_backend`
restaura la implementación C. La ruta MPS actual es síncrona y copia cada operando, aunque reutiliza las
asignaciones Metal por forma. Sirve como integración funcional y benchmark, no
como residencia completa del modelo en GPU ni como implementación de
`MPSGraph`.

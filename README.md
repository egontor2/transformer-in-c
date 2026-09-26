# transformer-in-c

Vision Transformer (ViT) escrito desde cero en C99, sin librerías de deep
learning, con entrenamiento completo en GPU mediante un backend Metal (Apple
Silicon) o CUDA (NVIDIA). El modelo se entrena y evalúa sobre
[OrganSMNIST](https://medmnist.com/) (MedMNIST v2): 11 órganos en cortes
sagitales de TAC abdominal de 28×28 píxeles.

Todo el pipeline es propio: tensores, autodiferenciación manual de cada capa,
AdamW, aumentación de datos, scheduler, checkpoints, evaluación con ensemble y
test-time augmentation, y los kernels de GPU. La implementación en C sirve
además como referencia numérica: cada kernel de GPU se valida contra ella.

## Resultados

Datos oficiales de MedMNIST (13 932 / 2 452 / 8 827 imágenes). Los modelos y
los hiperparámetros se eligen por validación; test solo se usa para informar.

| Modelo | Parámetros | Test acc. | Acc. equilibrada | Macro-F1 |
|---|---:|---:|---:|---:|
| ViT-C, patch 4, d=128, 4 capas | 0.80 M | 75.9 % | 0.718 | 0.721 |
| Ensemble de 2 ViT-C + TTA 9 vistas | 2 × 0.80 M | **78.6 %** | **0.745** | **0.754** |
| ResNet-18 (28×28), referencia MedMNIST v2 | 11 M | 78.2 % | n.d. | n.d. |

Cada época de entrenamiento tarda entre 8 y 13 s en un MacBook con chip M3
(GPU integrada). Casi un tercio de los errores se concentra en los pares fémur
izquierdo y derecho, y riñón izquierdo y derecho, que en un corte sagital no se
distinguen por el contenido de la imagen. La memoria en
[`paper/`](paper/) documenta el proyecto, los experimentos y la comparación
con modelos preentrenados en PyTorch.

## Características

**Modelo**
- ViT Pre-LN: patch embedding, token `[CLS]`, posiciones aprendibles,
  atención multi-cabeza bidireccional, MLP con GELU y LayerNorm final.
- Backward escrito a mano para cada operación, verificado con diferencias
  finitas.
- AdamW con weight decay desacoplado, clipping global por norma y label
  smoothing.

**Entrenamiento y evaluación**
- Warmup lineal y schedulers `cosine`, `plateau` y `constant`; early stopping
  y reanudación completa (pesos, momentos de AdamW y estado del scheduler).
- Normalización con estadísticas de entrenamiento.
- Aumentación que conserva la orientación anatómica (sin volteos):
  traslación, rotación, escala, brillo, contraste, ruido gaussiano y random
  erasing.
- Pesos por clase para datasets desbalanceados.
- Ensemble de checkpoints (con arquitecturas distintas) y TTA con 1, 5 o 9
  vistas desplazadas.
- Métricas por clase, precisión equilibrada, macro-F1 y matriz de confusión.

**Aceleración**
- CPU: implementación de referencia; en macOS las operaciones se paralelizan
  con GCD.
- Metal (macOS): el paso de entrenamiento completo se encola en la GPU y se
  sincroniza una sola vez por batch. GEMM con Metal Performance Shaders;
  el resto con kernels propios (`src/metal_kernels.metal`), incluida una
  atención por bloques tipo FlashAttention para secuencias largas.
- CUDA (NVIDIA): el mismo diseño con cuBLAS y kernels CUDA
  (`src/cuda_backend.cu`). Véase el [estado del backend CUDA](#estado-del-backend-cuda).

## Estructura

```text
include/            API pública (tensor.h, ops.h, vit.h, augmentation.h, ...)
src/
  autodiff.c        tensores, parámetros y allocator configurable
  ops.c             operaciones con implementación CPU y despacho a GPU
  vit.c             modelo ViT, entrenamiento, checkpoints y evaluación
  augmentation.c    aumentación de datos
  dataset.c         cargador de manifiestos CSV con imágenes PGM
  mps_backend.mm    backend Metal/MPS
  metal_kernels.metal
  cuda_backend.cu   backend CUDA/cuBLAS
examples/
  organ_smnist.c    entrenamiento y evaluación sobre OrganSMNIST
tests/
  test_vision.c     tests de la ruta CPU (incluye gradient checking)
  test_device.c     compara CPU y GPU (Metal o CUDA) paso a paso
scripts/            conversión de OrganSMNIST a PGM y análisis de datos
paper/              memoria del proyecto (LaTeX)
```

## Compilación

Requisitos: un compilador C99 y `make`. Para Metal, macOS con Xcode Command
Line Tools. Para CUDA, el CUDA Toolkit (11.5 o superior) y una GPU NVIDIA con
memoria gestionada (Pascal o posterior).

```sh
make                          # librería, ejemplo CPU y ejemplo básico
make test                     # tests de la ruta CPU

make build/organ_smnist_mps   # macOS: entrenamiento en GPU con Metal
make metal-test               # compara CPU y Metal

make build/organ_smnist_cuda  # Linux + NVIDIA: entrenamiento con CUDA
make cuda-test                # compara CPU y CUDA
```

Para CUDA, `CUDA_HOME` (por defecto `/usr/local/cuda`) y `CUDA_ARCH` (por
defecto `native`) pueden ajustarse, por ejemplo
`make build/organ_smnist_cuda CUDA_ARCH=sm_86`.

## Datos

Usa el fichero oficial `organsmnist.npz`, que ya contiene las imágenes a
28×28. El conversor genera imágenes PGM y los manifiestos `train.csv`,
`val.csv` y `test.csv`:

```sh
python3 -m pip install numpy
curl -L -o organsmnist.npz \
  "https://zenodo.org/records/10519652/files/organsmnist.npz?download=1"
python3 scripts/organ_smnist_to_pgm.py organsmnist.npz data/organ_smnist
```

`scripts/images_to_pgm.py` convierte copias del dataset distribuidas como
imágenes en carpetas por clase (requiere Pillow). Si esas imágenes están en
alta resolución, la reducción a 28×28 da píxeles distintos de los oficiales y
resultados no comparables: con la misma configuración, una copia reducida con
Pillow obtuvo 2.5 puntos más en test que los datos oficiales.

## Entrenamiento

Configuración con la que se obtienen los resultados de la tabla (sustituye
`organ_smnist_mps` por `organ_smnist_cuda` u `organ_smnist` según el backend):

```sh
./build/organ_smnist_mps \
  --train data/organ_smnist/train.csv \
  --val data/organ_smnist/val.csv \
  --test data/organ_smnist/test.csv \
  --checkpoint build/vit.ckpt \
  --batch 32 --epochs 50 --lr 0.001 --weight-decay 0.05 \
  --warmup 5 --scheduler cosine --early-stopping 15 \
  --patch-size 4 --d-model 128 --heads 8 --layers 4 \
  --augment standard --label-smoothing 0.1 \
  --metrics-csv build/vit.metrics.csv
```

El checkpoint guarda el mejor modelo según la pérdida de validación. Al
terminar se evalúa sobre test y se imprimen las métricas por clase y la matriz
de confusión.

| Opción | Descripción |
|---|---|
| `--patch-size`, `--d-model`, `--heads`, `--layers` | Arquitectura (patch_size debe dividir 28; d_model, ser múltiplo de heads) |
| `--batch`, `--epochs`, `--lr`, `--weight-decay` | Optimización |
| `--warmup N`, `--scheduler plateau\|cosine\|constant`, `--patience`, `--factor` | Planificación del learning rate |
| `--early-stopping N` | Parar tras N épocas sin mejorar la pérdida de validación (0 lo desactiva) |
| `--resume CKPT`, `--resume-lr VALUE` | Reanudar desde un checkpoint |
| `--augment standard` | Preset de aumentación; cada parámetro se ajusta con `--max-shift`, `--max-rotation`, `--max-scale`, `--brightness`, `--contrast`, `--noise-std`, `--erasing-prob`, `--erasing-size` |
| `--label-smoothing VALUE` | Label smoothing en la pérdida de entrenamiento |
| `--class-weights balanced` | Pesos inversamente proporcionales a la frecuencia de clase |
| `--seed N` | Semilla de inicialización, barajado y aumentación |
| `--metrics-csv PATH` | Métricas por época |

## Evaluación: ensemble y TTA

`--ensemble` evalúa checkpoints sin entrenar y promedia sus probabilidades.
Cada checkpoint guarda su arquitectura, así que se pueden mezclar modelos de
tamaños distintos. `--tta 5` añade las cuatro traslaciones de un píxel de cada
imagen; `--tta 9`, también las diagonales.

```sh
./build/organ_smnist_mps \
  --ensemble build/vit_seed1.ckpt,build/vit_seed2.ckpt --tta 9
```

Se informan las métricas de validación y test de cada modelo y del ensemble.
Elige la combinación por validación.

## Aceleración por GPU

Los dos backends siguen el mismo diseño:

1. **Memoria compartida.** Al activar el backend, `tensor_init` reserva cada
   tensor (pesos, activaciones, gradientes y momentos de AdamW) en memoria
   accesible desde CPU y GPU: `MTLBuffer` compartido en Metal y
   `cudaMallocManaged` en CUDA.
2. **Despacho por operación.** Cada función de `ops.h` consulta el
   `OpsDeviceBackend` instalado. Si todos sus tensores están en memoria de
   dispositivo, encola el kernel sin esperar; si no, sincroniza y ejecuta la
   versión C, que es la referencia.
3. **Una sincronización por paso.** Un paso de entrenamiento completo
   (patches, embeddings, bloques, pérdida, backward, clipping y AdamW) se
   encola entero y solo se sincroniza al leer la pérdida.

| | Metal | CUDA |
|---|---|---|
| GEMM | `MPSMatrixMultiplication` | cuBLAS `cublasSgemm` |
| Cola | command buffer pendiente | `cudaStream_t` |
| Reducciones | SIMD-groups de 32 hilos | warps |
| Parámetros agrupados | tabla de `gpuAddress` | tabla de punteros |

La atención elige entre tres kernels según el tamaño: uno con toda la
secuencia en memoria compartida (patch 4, 50 tokens), uno por bloques tipo
FlashAttention (patch 2, 197 tokens) y uno genérico para dimensiones de
cabeza mayores de 64.

### Estado del backend CUDA

El backend CUDA replica el de Metal kernel a kernel y se ha validado en una
NVIDIA L40S: `make cuda-test` pasa en los tres caminos de la atención y, con
los mismos datos y la misma semilla, el entrenamiento en la L40S y en un M3
con Metal coincide en pérdida y precisión de validación hasta la sexta cifra
decimal durante las dos primeras épocas.

## Tests

```sh
make test         # ruta CPU: operaciones, gradient checking, entrenamiento
make metal-test   # CPU frente a Metal
make cuda-test    # CPU frente a CUDA
make mps-test     # GEMM de MPS frente a CPU
```

`test_device.c` cubre los tres caminos de la atención (memoria compartida,
por bloques y genérico) y compara pérdida, gradientes y trayectoria de
entrenamiento con la implementación C.

## Limitaciones

- Solo precisión simple (fp32).
- Los datos se cargan desde PGM; no hay lector de `.npz` en C.
- El entrenamiento es de un solo proceso y una sola GPU.

## Licencia

Distribuido bajo la licencia MIT. Véase [`LICENSE`](LICENSE).

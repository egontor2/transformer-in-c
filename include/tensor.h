#ifndef TENSOR_H
#define TENSOR_H

#include "arena.h"

#include <stddef.h>

#define TENSOR_MAX_DIMS 4

typedef void *(*TensorAllocateFunction)(void *context, size_t bytes);
typedef void (*TensorReleaseFunction)(void *context, void *memory);

typedef struct {
    float *data;
    float *grad;
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t strides[TENSOR_MAX_DIMS];
    int owns_memory;
    TensorReleaseFunction release_memory;
    void *release_context;
    /* Convenience aliases for the common 2D case. */
    size_t rows;
    size_t cols;
} Tensor;

typedef struct {
    Tensor value;
    Tensor first_moment;
    Tensor second_moment;
    float *m;
    float *v;
    size_t step;
} Parameter;

void tensor_set_allocator(TensorAllocateFunction allocate,
                          TensorReleaseFunction release, void *context);

int tensor_init(Tensor *tensor, size_t rows, size_t cols);
int tensor_init_arena(Tensor *tensor, Arena *arena, size_t ndim,
                      const size_t *shape, int with_grad);
void tensor_free(Tensor *tensor);
void tensor_zero_grad(Tensor *tensor);
size_t tensor_numel(const Tensor *tensor);

int parameter_init(Parameter *parameter, size_t rows, size_t cols);
void parameter_free(Parameter *parameter);
void parameter_zero_grad(Parameter *parameter);

#endif

#include "autodiff.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int tensor_init(Tensor *tensor, size_t rows, size_t cols) {
    if (!tensor || rows == 0 || cols == 0) {
        return -1;
    }
    tensor->data = calloc(rows * cols, sizeof(float));
    tensor->grad = calloc(rows * cols, sizeof(float));
    tensor->ndim = 2;
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->strides[0] = cols;
    tensor->strides[1] = 1;
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->owns_memory = 1;
    if (!tensor->data || !tensor->grad) {
        tensor_free(tensor);
        return -1;
    }
    return 0;
}

int tensor_init_arena(Tensor *tensor, Arena *arena, size_t ndim,
                      const size_t *shape, int with_grad) {
    if (!tensor || !arena || !shape || ndim == 0 || ndim > TENSOR_MAX_DIMS) {
        return -1;
    }
    size_t count = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0 || count > SIZE_MAX / shape[i]) {
            return -1;
        }
        count *= shape[i];
    }
    memset(tensor, 0, sizeof(*tensor));
    tensor->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = i + 1 < ndim ? shape[i + 1] : 1;
        for (size_t j = i + 2; j < ndim; ++j) {
            tensor->strides[i] *= shape[j];
        }
    }
    tensor->rows = ndim == 2 ? shape[0] : 0;
    tensor->cols = ndim == 2 ? shape[1] : 0;
    tensor->owns_memory = 0;
    tensor->data = arena_alloc(arena, count * sizeof(float), sizeof(void *));
    tensor->grad = with_grad
        ? arena_alloc(arena, count * sizeof(float), sizeof(void *))
        : NULL;
    if (!tensor->data || (with_grad && !tensor->grad)) {
        return -1;
    }
    memset(tensor->data, 0, count * sizeof(float));
    if (tensor->grad) {
        memset(tensor->grad, 0, count * sizeof(float));
    }
    return 0;
}

void tensor_free(Tensor *tensor) {
    if (!tensor) {
        return;
    }
    if (tensor->owns_memory) {
        free(tensor->data);
        free(tensor->grad);
    }
    memset(tensor, 0, sizeof(*tensor));
}

size_t tensor_numel(const Tensor *tensor) {
    if (!tensor || tensor->ndim == 0) {
        return 0;
    }
    size_t count = 1;
    for (size_t i = 0; i < tensor->ndim; ++i) {
        count *= tensor->shape[i];
    }
    return count;
}

int parameter_init(Parameter *parameter, size_t rows, size_t cols) {
    if (!parameter || tensor_init(&parameter->value, rows, cols) != 0) {
        return -1;
    }
    const size_t count = rows * cols;
    parameter->m = calloc(count, sizeof(float));
    parameter->v = calloc(count, sizeof(float));
    parameter->step = 0;
    if (!parameter->m || !parameter->v) {
        parameter_free(parameter);
        return -1;
    }
    return 0;
}

void parameter_free(Parameter *parameter) {
    if (!parameter) {
        return;
    }
    tensor_free(&parameter->value);
    free(parameter->m);
    free(parameter->v);
    parameter->m = NULL;
    parameter->v = NULL;
    parameter->step = 0;
}

void parameter_zero_grad(Parameter *parameter) {
    if (parameter) {
        tensor_zero_grad(&parameter->value);
    }
}

void tensor_zero_grad(Tensor *tensor) {
    if (tensor && tensor->grad) {
        memset(tensor->grad, 0, tensor_numel(tensor) * sizeof(float));
    }
}

void tensor_matmul(const Tensor *left, const Tensor *right, Tensor *output) {
    if (!left || !right || !output || left->cols != right->rows ||
        output->rows != left->rows || output->cols != right->cols) {
        return;
    }
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t col = 0; col < output->cols; ++col) {
            float value = 0.0f;
            for (size_t i = 0; i < left->cols; ++i) {
                value += left->data[row * left->cols + i] *
                         right->data[i * right->cols + col];
            }
            output->data[row * output->cols + col] = value;
        }
    }
}

void tensor_matmul_backward(Tensor *left, Tensor *right, const Tensor *output) {
    if (!left || !right || !output || left->cols != right->rows ||
        output->rows != left->rows || output->cols != right->cols) {
        return;
    }
    for (size_t row = 0; row < left->rows; ++row) {
        for (size_t i = 0; i < left->cols; ++i) {
            for (size_t col = 0; col < right->cols; ++col) {
                left->grad[row * left->cols + i] +=
                    output->grad[row * output->cols + col] *
                    right->data[i * right->cols + col];
                right->grad[i * right->cols + col] +=
                    left->data[row * left->cols + i] *
                    output->grad[row * output->cols + col];
            }
        }
    }
}

void tensor_relu(Tensor *input, Tensor *output) {
    if (!input || !output || input->rows != output->rows ||
        input->cols != output->cols) {
        return;
    }
    const size_t count = input->rows * input->cols;
    for (size_t i = 0; i < count; ++i) {
        output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }
}

void tensor_relu_backward(Tensor *input, const Tensor *output) {
    if (!input || !output || input->rows != output->rows ||
        input->cols != output->cols) {
        return;
    }
    const size_t count = input->rows * input->cols;
    for (size_t i = 0; i < count; ++i) {
        if (input->data[i] > 0.0f) {
            input->grad[i] += output->grad[i];
        }
    }
}

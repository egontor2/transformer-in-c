#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "tensor.h"

int tensor_init(Tensor *tensor, size_t rows, size_t cols);
void tensor_free(Tensor *tensor);
void tensor_zero_grad(Tensor *tensor);

void tensor_matmul(const Tensor *left, const Tensor *right, Tensor *output);
void tensor_matmul_backward(Tensor *left, Tensor *right, const Tensor *output);
void tensor_relu(Tensor *input, Tensor *output);
void tensor_relu_backward(Tensor *input, const Tensor *output);

#endif

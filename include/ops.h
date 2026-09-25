#ifndef OPS_H
#define OPS_H

#include "tensor.h"

int ops_gemm(const Tensor *left, const Tensor *right, Tensor *output);
int ops_gemm_backward(Tensor *left, Tensor *right, const Tensor *output);
int ops_gemm_transposed_left(const Tensor *left, const Tensor *right,
                             Tensor *output);
int ops_gemm_transposed_left_backward(Tensor *left, Tensor *right,
                                      const Tensor *output);
int ops_residual(const Tensor *left, const Tensor *right, Tensor *output);
int ops_residual_backward(Tensor *left, Tensor *right, const Tensor *output);

int ops_layer_norm(const Tensor *input, const Tensor *gamma, const Tensor *beta,
                   float epsilon, Tensor *output);
int ops_layer_norm_backward(const Tensor *input, const Tensor *gamma,
                            float epsilon, const Tensor *output,
                            Tensor *input_grad, Tensor *gamma_grad,
                            Tensor *beta_grad);
int ops_gelu(const Tensor *input, Tensor *output);
int ops_gelu_backward(const Tensor *input, const Tensor *output);
int ops_softmax_cross_entropy(const Tensor *logits, const size_t *targets,
                             float *loss, Tensor *logits_grad);
int ops_adamw_step(Parameter *parameter, float learning_rate,
                   float beta1, float beta2, float epsilon,
                   float weight_decay);

#endif

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

typedef int (*OpsGemmBackend)(void *context, const float *left,
                              const float *right, float *output,
                              size_t left_rows, size_t left_cols,
                              size_t right_cols);
typedef int (*OpsGemmBackwardBackend)(
    void *context, const float *left, const float *right,
    const float *output_grad, float *left_grad, float *right_grad,
    size_t left_rows, size_t left_cols, size_t right_cols);

/* The backend is process-global; install it only around single-threaded work. */
void ops_set_gemm_backend(OpsGemmBackend backend, void *context);
void ops_set_gemm_backward_backend(OpsGemmBackwardBackend backend,
                                   void *context);
void ops_reset_gemm_backend(void);
int ops_gemm(const Tensor *left, const Tensor *right, Tensor *output);
int ops_gemm_backward(Tensor *left, Tensor *right, const Tensor *output);
int ops_gemm_transposed_left(const Tensor *left, const Tensor *right,
                             Tensor *output);
int ops_gemm_transposed_left_backward(Tensor *left, Tensor *right,
                                      const Tensor *output);
int ops_residual(const Tensor *left, const Tensor *right, Tensor *output);
int ops_residual_backward(Tensor *left, Tensor *right, const Tensor *output);
int ops_bias_add(const Tensor *input, const Tensor *bias, Tensor *output);
int ops_bias_add_backward(const Tensor *output, Tensor *bias_grad);

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

int ops_causal_attention(const Tensor *query, const Tensor *key,
                         const Tensor *value, float scale,
                         Tensor *probabilities, Tensor *output);
int ops_causal_attention_backward(const Tensor *query, const Tensor *key,
                                  const Tensor *value, float scale,
                                  const Tensor *probabilities,
                                  const Tensor *output, Tensor *query_grad,
                                  Tensor *key_grad, Tensor *value_grad);
int ops_multi_head_attention(const Tensor *query, const Tensor *key,
                             const Tensor *value, size_t batch,
                             size_t heads, size_t sequence_length,
                             float scale, Tensor *probabilities,
                             Tensor *output);
int ops_multi_head_attention_backward(
    const Tensor *query, const Tensor *key, const Tensor *value,
    size_t batch, size_t heads, size_t sequence_length, float scale,
    const Tensor *probabilities, const Tensor *output, Tensor *query_grad,
    Tensor *key_grad, Tensor *value_grad);

#endif

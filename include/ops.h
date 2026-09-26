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

typedef struct {
    void *context;
    int (*gemm)(void *context, const Tensor *left, const Tensor *right,
                Tensor *output);
    int (*gemm_backward)(void *context, Tensor *left, Tensor *right,
                         const Tensor *output);
    int (*residual)(void *context, const Tensor *left, const Tensor *right,
                    Tensor *output);
    int (*residual_backward)(void *context, Tensor *left, Tensor *right,
                             const Tensor *output);
    int (*accumulate_gradient)(void *context, Tensor *destination,
                               const Tensor *source);
    int (*bias_add)(void *context, const Tensor *input, const Tensor *bias,
                    Tensor *output);
    int (*bias_add_backward)(void *context, const Tensor *output,
                             Tensor *bias_grad);
    int (*layer_norm)(void *context, const Tensor *input, const Tensor *gamma,
                      const Tensor *beta, float epsilon, Tensor *output);
    int (*layer_norm_backward)(void *context, const Tensor *input,
                               const Tensor *gamma, float epsilon,
                               const Tensor *output, Tensor *input_grad,
                               Tensor *gamma_grad, Tensor *beta_grad);
    int (*gelu)(void *context, const Tensor *input, Tensor *output);
    int (*gelu_backward)(void *context, const Tensor *input,
                         const Tensor *output);
    int (*split_heads)(void *context, const Tensor *projected, Tensor *query,
                       Tensor *key, Tensor *value, size_t batch,
                       size_t sequence_length, size_t d_model, size_t heads);
    int (*split_heads_backward)(void *context, Tensor *projected,
                                const Tensor *query, const Tensor *key,
                                const Tensor *value, size_t batch,
                                size_t sequence_length, size_t d_model,
                                size_t heads);
    int (*merge_heads)(void *context, const Tensor *attended, Tensor *merged,
                       size_t batch, size_t sequence_length, size_t d_model,
                       size_t heads);
    int (*merge_heads_backward)(void *context, Tensor *attended,
                                const Tensor *merged, size_t batch,
                                size_t sequence_length, size_t d_model,
                                size_t heads);
    int (*multi_head_attention)(void *context, const Tensor *query,
                                const Tensor *key, const Tensor *value,
                                size_t batch, size_t heads,
                                size_t sequence_length, float scale,
                                Tensor *probabilities, Tensor *output);
    int (*multi_head_attention_backward)(
        void *context, const Tensor *query, const Tensor *key,
        const Tensor *value, size_t batch, size_t heads,
        size_t sequence_length, float scale, const Tensor *probabilities,
        const Tensor *output, Tensor *query_grad, Tensor *key_grad,
        Tensor *value_grad);
    int (*zero_gradient)(void *context, Tensor *tensor);
    int (*extract_patches)(void *context, const Tensor *images,
                           Tensor *patches, size_t channels, size_t height,
                           size_t width, size_t patch_size);
    int (*token_embedding)(void *context, const Tensor *patch_tokens,
                           const Tensor *cls_token, const Tensor *positional,
                           Tensor *tokens, size_t batch);
    int (*token_embedding_backward)(void *context, Tensor *patch_tokens,
                                    Tensor *cls_token, Tensor *positional,
                                    const Tensor *tokens, size_t batch);
    int (*gather_rows)(void *context, const Tensor *source,
                       Tensor *destination, size_t row_stride);
    int (*gather_rows_backward)(void *context, Tensor *source,
                                const Tensor *destination, size_t row_stride);
    int (*softmax_cross_entropy)(void *context, const Tensor *logits,
                                 const Tensor *targets,
                                 const Tensor *class_weights,
                                 float label_smoothing, Tensor *loss,
                                 Tensor *logits_grad);
    int (*clip_gradient_norm)(void *context, Parameter *const *parameters,
                              size_t parameter_count, float max_norm);
    int (*adamw_step)(void *context, Parameter *parameter,
                      float learning_rate, float beta1, float beta2,
                      float epsilon, float weight_decay, float correction1,
                      float correction2);
    int (*zero_gradients)(void *context, Tensor *const *tensors,
                          size_t tensor_count);
    int (*adamw_step_parameters)(void *context, Parameter *const *parameters,
                                 const float *weight_decays,
                                 size_t parameter_count, float learning_rate,
                                 float beta1, float beta2, float epsilon,
                                 float max_gradient_norm);
    void (*synchronize)(void *context);
} OpsDeviceBackend;

void ops_set_device_backend(const OpsDeviceBackend *backend);
void ops_synchronize(void);
int ops_gemm(const Tensor *left, const Tensor *right, Tensor *output);
int ops_gemm_backward(Tensor *left, Tensor *right, const Tensor *output);
int ops_gemm_transposed_left(const Tensor *left, const Tensor *right,
                             Tensor *output);
int ops_gemm_transposed_left_backward(Tensor *left, Tensor *right,
                                      const Tensor *output);
int ops_residual(const Tensor *left, const Tensor *right, Tensor *output);
int ops_residual_backward(Tensor *left, Tensor *right, const Tensor *output);
int ops_accumulate_gradient(Tensor *destination, const Tensor *source);
int ops_zero_gradient(Tensor *tensor);
int ops_zero_gradients(Tensor *const *tensors, size_t tensor_count);
int ops_upload_values(Tensor *tensor, const float *values);
int ops_upload_labels(Tensor *tensor, const size_t *labels);
int ops_extract_patches(const Tensor *images, Tensor *patches, size_t channels,
                        size_t height, size_t width, size_t patch_size);
int ops_extract_patches_backward(const Tensor *patches, float *image_grad,
                                 size_t channels, size_t height, size_t width,
                                 size_t patch_size);
int ops_token_embedding(const Tensor *patch_tokens, const Tensor *cls_token,
                        const Tensor *positional, Tensor *tokens,
                        size_t batch);
int ops_token_embedding_backward(Tensor *patch_tokens, Tensor *cls_token,
                                 Tensor *positional, const Tensor *tokens,
                                 size_t batch);
int ops_gather_rows(const Tensor *source, Tensor *destination,
                    size_t row_stride);
int ops_gather_rows_backward(Tensor *source, const Tensor *destination,
                             size_t row_stride);
int ops_softmax_cross_entropy_tensor(const Tensor *logits,
                                     const Tensor *targets,
                                     const Tensor *class_weights,
                                     float label_smoothing, Tensor *loss,
                                     Tensor *logits_grad);
int ops_clip_gradient_norm(Parameter *const *parameters,
                           size_t parameter_count, float max_norm);
int ops_split_heads(const Tensor *projected, Tensor *query, Tensor *key,
                    Tensor *value, size_t batch, size_t sequence_length,
                    size_t d_model, size_t heads);
int ops_split_heads_backward(Tensor *projected, const Tensor *query,
                             const Tensor *key, const Tensor *value,
                             size_t batch, size_t sequence_length,
                             size_t d_model, size_t heads);
int ops_merge_heads(const Tensor *attended, Tensor *merged, size_t batch,
                    size_t sequence_length, size_t d_model, size_t heads);
int ops_merge_heads_backward(Tensor *attended, const Tensor *merged,
                             size_t batch, size_t sequence_length,
                             size_t d_model, size_t heads);
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
int ops_softmax_cross_entropy_smoothed(const Tensor *logits,
                                       const size_t *targets,
                                       const float *class_weights,
                                       float label_smoothing, float *loss,
                                       Tensor *logits_grad);
int ops_softmax_cross_entropy_weighted(const Tensor *logits,
                                       const size_t *targets,
                                       const float *class_weights,
                                       float *loss, Tensor *logits_grad);
int ops_adamw_step(Parameter *parameter, float learning_rate,
                   float beta1, float beta2, float epsilon,
                   float weight_decay);
int ops_adamw_step_parameters(Parameter *const *parameters,
                              const float *weight_decays,
                              size_t parameter_count, float learning_rate,
                              float beta1, float beta2, float epsilon,
                              float max_gradient_norm);

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

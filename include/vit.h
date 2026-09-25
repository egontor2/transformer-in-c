#ifndef VIT_H
#define VIT_H

#include "tensor.h"
#include "ops.h"

#include <stddef.h>

typedef struct {
    size_t channels;
    size_t height;
    size_t width;
    size_t patch_size;
    size_t d_model;
    size_t patch_count;
    Parameter projection;
    Parameter bias;
} ViTPatchProjection;

int vit_patch_projection_init(ViTPatchProjection *projection,
                              size_t channels, size_t height, size_t width,
                              size_t patch_size, size_t d_model);
void vit_patch_projection_free(ViTPatchProjection *projection);

/*
 * Input layout is [batch][channels][height][width].
 * Output layout is [batch * patch_count][d_model].
 */
int vit_patch_projection_forward(const ViTPatchProjection *projection,
                                 const float *input, size_t batch,
                                 Tensor *output);

/*
 * Accumulates gradients into projection parameters and input_grad.
 * input_grad must use the same [batch][channels][height][width] layout.
 */
int vit_patch_projection_backward(ViTPatchProjection *projection,
                                  const float *input, size_t batch,
                                  const Tensor *output, float *input_grad);

typedef struct {
    size_t d_model;
    size_t patch_count;
    Parameter cls_token;
    Parameter positional_embeddings;
} ViTTokenEmbedding;

int vit_token_embedding_init(ViTTokenEmbedding *embedding,
                             size_t patch_count, size_t d_model);
void vit_token_embedding_free(ViTTokenEmbedding *embedding);
int vit_token_embedding_forward(const ViTTokenEmbedding *embedding,
                                const Tensor *patch_tokens, size_t batch,
                                Tensor *output);
int vit_token_embedding_backward(ViTTokenEmbedding *embedding,
                                 const Tensor *output, size_t batch,
                                 Tensor *patch_grad);

typedef struct {
    size_t d_model;
    size_t heads;
    size_t sequence_length;
    Parameter query_key_value;
    Parameter query_key_value_bias;
    Parameter attention_output;
    Parameter attention_output_bias;
    Parameter mlp_input;
    Parameter mlp_input_bias;
    Parameter mlp_output;
    Parameter mlp_output_bias;
    Parameter attention_gamma;
    Parameter attention_beta;
    Parameter mlp_gamma;
    Parameter mlp_beta;
} ViTEncoderBlock;

typedef struct {
    Tensor normalized_attention;
    Tensor projected;
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor probabilities;
    Tensor attended;
    Tensor merged;
    Tensor attention_branch;
    Tensor attention_residual;
    Tensor normalized_mlp;
    Tensor hidden;
    Tensor activated;
    Tensor mlp_branch;
    Tensor output;
    size_t batch;
} ViTEncoderBlockCache;

int vit_encoder_block_init(ViTEncoderBlock *block, size_t d_model,
                           size_t heads, size_t sequence_length);
void vit_encoder_block_free(ViTEncoderBlock *block);
int vit_encoder_block_forward(const ViTEncoderBlock *block,
                              const Tensor *input, size_t batch,
                              Tensor *output);
int vit_encoder_block_cache_init(ViTEncoderBlockCache *cache,
                                 const ViTEncoderBlock *block, size_t batch);
void vit_encoder_block_cache_free(ViTEncoderBlockCache *cache);
int vit_encoder_block_forward_cached(const ViTEncoderBlock *block,
                                     const Tensor *input,
                                     ViTEncoderBlockCache *cache);
int vit_encoder_block_backward(ViTEncoderBlock *block,
                               const Tensor *input,
                               ViTEncoderBlockCache *cache,
                               Tensor *input_grad);

typedef struct {
    size_t d_model;
    size_t classes;
    Parameter projection;
    Parameter bias;
} ViTClassificationHead;

int vit_classification_head_init(ViTClassificationHead *head, size_t d_model,
                                 size_t classes);
void vit_classification_head_free(ViTClassificationHead *head);
int vit_classification_head_forward(const ViTClassificationHead *head,
                                    const Tensor *tokens, size_t batch,
                                    size_t sequence_length, Tensor *logits);
int vit_classification_head_backward(ViTClassificationHead *head,
                                     const Tensor *tokens, size_t batch,
                                     size_t sequence_length,
                                     const Tensor *logits, Tensor *token_grad);

typedef struct {
    size_t channels;
    size_t height;
    size_t width;
    size_t patch_size;
    size_t d_model;
    size_t heads;
    size_t layers;
    size_t classes;
} ViTConfig;

typedef struct {
    ViTConfig config;
    size_t patch_count;
    size_t sequence_length;
    ViTPatchProjection patch_projection;
    ViTTokenEmbedding token_embedding;
    ViTEncoderBlock *blocks;
    Parameter final_gamma;
    Parameter final_beta;
    ViTClassificationHead classification_head;
} ViTModel;

typedef struct {
    Tensor patch_tokens;
    Tensor tokens;
    ViTEncoderBlockCache *block_caches;
    Tensor normalized_tokens;
    Tensor logits;
    size_t batch;
    size_t layers;
} ViTModelCache;

typedef struct {
    float accuracy;
    float average_loss;
    double elapsed_seconds;
    double samples_per_second;
} ViTEvaluationMetrics;

int vit_model_init(ViTModel *model, const ViTConfig *config);
void vit_model_free(ViTModel *model);
void vit_model_zero_grad(ViTModel *model);
int vit_model_cache_init(ViTModelCache *cache, const ViTModel *model,
                         size_t batch);
void vit_model_cache_free(ViTModelCache *cache);
int vit_model_clip_gradients(ViTModel *model, float max_norm);
int vit_model_forward(const ViTModel *model, const float *images, size_t batch,
                      ViTModelCache *cache);
int vit_model_backward(ViTModel *model, const float *images,
                       ViTModelCache *cache, float *image_grad);
int vit_model_train_batch(ViTModel *model, const float *images,
                          const size_t *targets, size_t batch,
                          ViTModelCache *cache, float learning_rate,
                          float weight_decay, float *loss);
int vit_model_train_batch_weighted(
    ViTModel *model, const float *images, const size_t *targets, size_t batch,
    ViTModelCache *cache, float learning_rate, float weight_decay,
    const float *class_weights, float *loss);
int vit_model_save(const ViTModel *model, const char *path);
int vit_model_load(ViTModel *model, const char *path);
int vit_model_evaluate(const ViTModel *model, const float *images,
                       const size_t *targets, size_t sample_count,
                       ViTEvaluationMetrics *metrics);

#endif

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
    Parameter attention_output;
    Parameter mlp_input;
    Parameter mlp_output;
    Parameter attention_gamma;
    Parameter attention_beta;
    Parameter mlp_gamma;
    Parameter mlp_beta;
} ViTEncoderBlock;

int vit_encoder_block_init(ViTEncoderBlock *block, size_t d_model,
                           size_t heads, size_t sequence_length);
void vit_encoder_block_free(ViTEncoderBlock *block);
int vit_encoder_block_forward(const ViTEncoderBlock *block,
                              const Tensor *input, size_t batch,
                              Tensor *output);

#endif

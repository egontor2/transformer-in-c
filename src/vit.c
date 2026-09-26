#define _POSIX_C_SOURCE 199309L

#include "vit.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int valid_configuration(size_t channels, size_t height, size_t width,
                               size_t patch_size, size_t d_model) {
    return channels > 0 && height > 0 && width > 0 && patch_size > 0 &&
           d_model > 0 && height % patch_size == 0 && width % patch_size == 0;
}

int vit_patch_projection_init(ViTPatchProjection *projection,
                              size_t channels, size_t height, size_t width,
                              size_t patch_size, size_t d_model) {
    if (!projection || !valid_configuration(channels, height, width,
                                             patch_size, d_model)) {
        return -1;
    }
    memset(projection, 0, sizeof(*projection));
    projection->channels = channels;
    projection->height = height;
    projection->width = width;
    projection->patch_size = patch_size;
    projection->d_model = d_model;
    projection->patch_count = (height / patch_size) * (width / patch_size);
    const size_t patch_elements = channels * patch_size * patch_size;
    if (parameter_init(&projection->projection, patch_elements, d_model) != 0 ||
        parameter_init(&projection->bias, 1, d_model) != 0) {
        vit_patch_projection_free(projection);
        return -1;
    }
    return 0;
}

void vit_patch_projection_free(ViTPatchProjection *projection) {
    if (!projection) {
        return;
    }
    parameter_free(&projection->projection);
    parameter_free(&projection->bias);
    memset(projection, 0, sizeof(*projection));
}

int vit_patch_projection_forward(const ViTPatchProjection *projection,
                                 const float *input, size_t batch,
                                 Tensor *output) {
    ops_synchronize();
    if (!projection || !input || !output || batch == 0 ||
        output->ndim != 2 ||
        output->rows != batch * projection->patch_count ||
        output->cols != projection->d_model) {
        return -1;
    }
    const size_t patch_size = projection->patch_size;
    const size_t patch_elements = projection->channels * patch_size * patch_size;
    const size_t image_elements = projection->channels * projection->height *
                                  projection->width;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t patch_y = 0; patch_y < projection->height; patch_y += patch_size) {
            for (size_t patch_x = 0; patch_x < projection->width; patch_x += patch_size) {
                const size_t patch_index =
                    (patch_y / patch_size) * (projection->width / patch_size) +
                    patch_x / patch_size;
                const size_t output_row = sample * projection->patch_count + patch_index;
                for (size_t dimension = 0; dimension < projection->d_model; ++dimension) {
                    float value = projection->bias.value.data[dimension];
                    size_t flattened = 0;
                    for (size_t channel = 0; channel < projection->channels; ++channel) {
                        for (size_t y = 0; y < patch_size; ++y) {
                            for (size_t x = 0; x < patch_size; ++x) {
                                const size_t image_index =
                                    sample * image_elements +
                                    channel * projection->height * projection->width +
                                    (patch_y + y) * projection->width + patch_x + x;
                                value += input[image_index] *
                                    projection->projection.value.data[
                                        flattened * projection->d_model + dimension];
                                ++flattened;
                            }
                        }
                    }
                    output->data[output_row * output->cols + dimension] = value;
                }
            }
        }
    }
    (void)patch_elements;
    return 0;
}

int vit_patch_projection_backward(ViTPatchProjection *projection,
                                  const float *input, size_t batch,
                                  const Tensor *output, float *input_grad) {
    ops_synchronize();
    if (!projection || !input || !output || !input_grad || batch == 0 ||
        output->ndim != 2 ||
        output->rows != batch * projection->patch_count ||
        output->cols != projection->d_model) {
        return -1;
    }
    const size_t patch_size = projection->patch_size;
    const size_t image_elements = projection->channels * projection->height *
                                  projection->width;
    const size_t patch_elements = projection->channels * patch_size * patch_size;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t patch_y = 0; patch_y < projection->height; patch_y += patch_size) {
            for (size_t patch_x = 0; patch_x < projection->width; patch_x += patch_size) {
                const size_t patch_index =
                    (patch_y / patch_size) * (projection->width / patch_size) +
                    patch_x / patch_size;
                const size_t output_row = sample * projection->patch_count + patch_index;
                for (size_t dimension = 0; dimension < projection->d_model; ++dimension) {
                    const float upstream =
                        output->grad[output_row * output->cols + dimension];
                    projection->bias.value.grad[dimension] += upstream;
                    size_t flattened = 0;
                    for (size_t channel = 0; channel < projection->channels; ++channel) {
                        for (size_t y = 0; y < patch_size; ++y) {
                            for (size_t x = 0; x < patch_size; ++x) {
                                const size_t image_index =
                                    sample * image_elements +
                                    channel * projection->height * projection->width +
                                    (patch_y + y) * projection->width + patch_x + x;
                                const size_t weight_index =
                                    flattened * projection->d_model + dimension;
                                projection->projection.value.grad[weight_index] +=
                                    input[image_index] * upstream;
                                input_grad[image_index] +=
                                    projection->projection.value.data[weight_index] * upstream;
                                ++flattened;
                            }
                        }
                    }
                }
            }
        }
    }
    (void)patch_elements;
    return 0;
}

int vit_token_embedding_init(ViTTokenEmbedding *embedding,
                             size_t patch_count, size_t d_model) {
    if (!embedding || patch_count == 0 || d_model == 0) {
        return -1;
    }
    memset(embedding, 0, sizeof(*embedding));
    embedding->patch_count = patch_count;
    embedding->d_model = d_model;
    if (parameter_init(&embedding->cls_token, 1, d_model) != 0 ||
        parameter_init(&embedding->positional_embeddings, patch_count + 1,
                       d_model) != 0) {
        vit_token_embedding_free(embedding);
        return -1;
    }
    return 0;
}

void vit_token_embedding_free(ViTTokenEmbedding *embedding) {
    if (!embedding) {
        return;
    }
    parameter_free(&embedding->cls_token);
    parameter_free(&embedding->positional_embeddings);
    memset(embedding, 0, sizeof(*embedding));
}

int vit_token_embedding_forward(const ViTTokenEmbedding *embedding,
                                const Tensor *patch_tokens, size_t batch,
                                Tensor *output) {
    ops_synchronize();
    if (!embedding || !patch_tokens || !output || batch == 0 ||
        patch_tokens->ndim != 2 || output->ndim != 2 ||
        patch_tokens->rows != batch * embedding->patch_count ||
        patch_tokens->cols != embedding->d_model ||
        output->rows != batch * (embedding->patch_count + 1) ||
        output->cols != embedding->d_model) {
        return -1;
    }
    const size_t sequence_length = embedding->patch_count + 1;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const size_t output_row = sample * sequence_length + position;
            for (size_t dimension = 0; dimension < embedding->d_model; ++dimension) {
                const float token = position == 0
                    ? embedding->cls_token.value.data[dimension]
                    : patch_tokens->data[
                        (sample * embedding->patch_count + position - 1) *
                        embedding->d_model + dimension];
                output->data[output_row * output->cols + dimension] =
                    token + embedding->positional_embeddings.value.data[
                        position * embedding->d_model + dimension];
            }
        }
    }
    return 0;
}

int vit_token_embedding_backward(ViTTokenEmbedding *embedding,
                                 const Tensor *output, size_t batch,
                                 Tensor *patch_grad) {
    ops_synchronize();
    if (!embedding || !output || !patch_grad || batch == 0 ||
        output->ndim != 2 || patch_grad->ndim != 2 ||
        output->rows != batch * (embedding->patch_count + 1) ||
        output->cols != embedding->d_model ||
        patch_grad->rows != batch * embedding->patch_count ||
        patch_grad->cols != embedding->d_model) {
        return -1;
    }
    const size_t sequence_length = embedding->patch_count + 1;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const size_t output_row = sample * sequence_length + position;
            for (size_t dimension = 0; dimension < embedding->d_model; ++dimension) {
                const float gradient =
                    output->grad[output_row * output->cols + dimension];
                embedding->positional_embeddings.value.grad[
                    position * embedding->d_model + dimension] += gradient;
                if (position == 0) {
                    embedding->cls_token.value.grad[dimension] += gradient;
                } else {
                    patch_grad->grad[
                        (sample * embedding->patch_count + position - 1) *
                        embedding->d_model + dimension] += gradient;
                }
            }
        }
    }
    return 0;
}

static int initialize_parameter(Parameter *parameter, size_t rows, size_t cols) {
    return parameter_init(parameter, rows, cols);
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t mixed = (*state += 0x9E3779B97F4A7C15ULL);
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

static float random_open_unit(uint64_t *state) {
    return ((float)(splitmix64_next(state) >> 40) + 0.5f) / 16777216.0f;
}

static float random_standard_normal(uint64_t *state) {
    const float two_pi = 6.28318530717958647692f;
    const float radius = sqrtf(-2.0f * logf(random_open_unit(state)));
    return radius * cosf(two_pi * random_open_unit(state));
}

static void initialize_xavier_uniform(Parameter *parameter, uint64_t *state) {
    const float fan_in = (float)parameter->value.rows;
    const float fan_out = (float)parameter->value.cols;
    const float limit = sqrtf(6.0f / (fan_in + fan_out));
    const size_t count = tensor_numel(&parameter->value);
    for (size_t index = 0; index < count; ++index) {
        parameter->value.data[index] =
            limit * (2.0f * random_open_unit(state) - 1.0f);
    }
}

static void initialize_truncated_normal(Parameter *parameter,
                                        float standard_deviation,
                                        uint64_t *state) {
    const float truncation_bound = 2.0f;
    const size_t count = tensor_numel(&parameter->value);
    for (size_t index = 0; index < count; ++index) {
        float sample = random_standard_normal(state);
        while (fabsf(sample) > truncation_bound) {
            sample = random_standard_normal(state);
        }
        parameter->value.data[index] = standard_deviation * sample;
    }
}

int vit_encoder_block_init(ViTEncoderBlock *block, size_t d_model,
                           size_t heads, size_t sequence_length) {
    if (!block || d_model == 0 || heads == 0 || sequence_length == 0 ||
        d_model % heads != 0) {
        return -1;
    }
    memset(block, 0, sizeof(*block));
    block->d_model = d_model;
    block->heads = heads;
    block->sequence_length = sequence_length;
    if (initialize_parameter(&block->query_key_value, d_model, 3 * d_model) ||
        initialize_parameter(&block->query_key_value_bias, 1, 3 * d_model) ||
        initialize_parameter(&block->attention_output, d_model, d_model) ||
        initialize_parameter(&block->attention_output_bias, 1, d_model) ||
        initialize_parameter(&block->mlp_input, d_model, 4 * d_model) ||
        initialize_parameter(&block->mlp_input_bias, 1, 4 * d_model) ||
        initialize_parameter(&block->mlp_output, 4 * d_model, d_model) ||
        initialize_parameter(&block->mlp_output_bias, 1, d_model) ||
        initialize_parameter(&block->attention_gamma, 1, d_model) ||
        initialize_parameter(&block->attention_beta, 1, d_model) ||
        initialize_parameter(&block->mlp_gamma, 1, d_model) ||
        initialize_parameter(&block->mlp_beta, 1, d_model)) {
        vit_encoder_block_free(block);
        return -1;
    }
    for (size_t i = 0; i < d_model; ++i) {
        block->attention_gamma.value.data[i] = 1.0f;
        block->mlp_gamma.value.data[i] = 1.0f;
    }
    return 0;
}

void vit_encoder_block_free(ViTEncoderBlock *block) {
    if (!block) {
        return;
    }
    parameter_free(&block->query_key_value);
    parameter_free(&block->query_key_value_bias);
    parameter_free(&block->attention_output);
    parameter_free(&block->attention_output_bias);
    parameter_free(&block->mlp_input);
    parameter_free(&block->mlp_input_bias);
    parameter_free(&block->mlp_output);
    parameter_free(&block->mlp_output_bias);
    parameter_free(&block->attention_gamma);
    parameter_free(&block->attention_beta);
    parameter_free(&block->mlp_gamma);
    parameter_free(&block->mlp_beta);
    memset(block, 0, sizeof(*block));
}

static int create_matrix(Tensor *tensor, size_t rows, size_t cols) {
    return tensor_init(tensor, rows, cols);
}

int vit_encoder_block_forward(const ViTEncoderBlock *block,
                              const Tensor *input, size_t batch,
                              Tensor *output) {
    if (!block || !input || !output || batch == 0 ||
        input->ndim != 2 || output->ndim != 2 ||
        input->rows != batch * block->sequence_length ||
        output->rows != input->rows || input->cols != block->d_model ||
        output->cols != block->d_model) {
        return -1;
    }
    const size_t rows = batch * block->sequence_length;
    const size_t head_dimension = block->d_model / block->heads;
    const size_t attention_rows = batch * block->heads * block->sequence_length;
    Tensor normalized_attention = {0};
    Tensor projected = {0};
    Tensor query = {0};
    Tensor key = {0};
    Tensor value = {0};
    Tensor probabilities = {0};
    Tensor attended = {0};
    Tensor merged = {0};
    Tensor attention_residual = {0};
    Tensor mlp_residual = {0};
    Tensor normalized_mlp = {0};
    Tensor hidden = {0};
    Tensor activated = {0};
    int result = create_matrix(&normalized_attention, rows, block->d_model) ||
        create_matrix(&projected, rows, 3 * block->d_model) ||
        create_matrix(&query, attention_rows, head_dimension) ||
        create_matrix(&key, attention_rows, head_dimension) ||
        create_matrix(&value, attention_rows, head_dimension) ||
        create_matrix(&probabilities, attention_rows, block->sequence_length) ||
        create_matrix(&attended, attention_rows, head_dimension) ||
        create_matrix(&merged, rows, block->d_model) ||
        create_matrix(&attention_residual, rows, block->d_model) ||
        create_matrix(&mlp_residual, rows, block->d_model) ||
        create_matrix(&normalized_mlp, rows, block->d_model) ||
        create_matrix(&hidden, rows, 4 * block->d_model) ||
        create_matrix(&activated, rows, 4 * block->d_model);
    if (result == 0) {
        result = ops_layer_norm(input, &block->attention_gamma.value,
                                &block->attention_beta.value, 1e-5f,
                                &normalized_attention) ||
            ops_gemm(&normalized_attention, &block->query_key_value.value,
                     &projected);
    }
    if (result == 0) {
        result = ops_split_heads(&projected, &query, &key, &value, batch,
                                 block->sequence_length, block->d_model,
                                 block->heads) ||
            ops_multi_head_attention(
            &query, &key, &value, batch, block->heads,
            block->sequence_length, 1.0f / sqrtf((float)head_dimension),
            &probabilities, &attended);
    }
    if (result == 0) {
        result = ops_merge_heads(&attended, &merged, batch,
                                 block->sequence_length, block->d_model,
                                 block->heads) ||
            ops_gemm(&merged, &block->attention_output.value,
                          &mlp_residual) ||
            ops_residual(input, &mlp_residual, &attention_residual) ||
            ops_layer_norm(&attention_residual, &block->mlp_gamma.value,
                           &block->mlp_beta.value, 1e-5f, &normalized_mlp) ||
            ops_gemm(&normalized_mlp, &block->mlp_input.value, &hidden) ||
            ops_gelu(&hidden, &activated) ||
            ops_gemm(&activated, &block->mlp_output.value, &mlp_residual) ||
            ops_residual(&attention_residual, &mlp_residual, output);
    }
    tensor_free(&normalized_attention);
    tensor_free(&projected);
    tensor_free(&query);
    tensor_free(&key);
    tensor_free(&value);
    tensor_free(&probabilities);
    tensor_free(&attended);
    tensor_free(&merged);
    tensor_free(&attention_residual);
    tensor_free(&mlp_residual);
    tensor_free(&normalized_mlp);
    tensor_free(&hidden);
    tensor_free(&activated);
    return result;
}

static void free_block_cache_tensors(ViTEncoderBlockCache *cache) {
    tensor_free(&cache->normalized_attention);
    tensor_free(&cache->projected);
    tensor_free(&cache->query);
    tensor_free(&cache->key);
    tensor_free(&cache->value);
    tensor_free(&cache->probabilities);
    tensor_free(&cache->attended);
    tensor_free(&cache->merged);
    tensor_free(&cache->attention_branch);
    tensor_free(&cache->attention_residual);
    tensor_free(&cache->normalized_mlp);
    tensor_free(&cache->hidden);
    tensor_free(&cache->activated);
    tensor_free(&cache->mlp_branch);
    tensor_free(&cache->output);
}

int vit_encoder_block_cache_init(ViTEncoderBlockCache *cache,
                                 const ViTEncoderBlock *block, size_t batch) {
    if (!cache || !block || batch == 0) {
        return -1;
    }
    memset(cache, 0, sizeof(*cache));
    cache->batch = batch;
    const size_t rows = batch * block->sequence_length;
    const size_t head_dimension = block->d_model / block->heads;
    const size_t attention_rows = rows * block->heads;
    int result = tensor_init(&cache->normalized_attention, rows, block->d_model) ||
        tensor_init(&cache->projected, rows, 3 * block->d_model) ||
        tensor_init(&cache->query, attention_rows, head_dimension) ||
        tensor_init(&cache->key, attention_rows, head_dimension) ||
        tensor_init(&cache->value, attention_rows, head_dimension) ||
        tensor_init(&cache->probabilities, attention_rows,
                    block->sequence_length) ||
        tensor_init(&cache->attended, attention_rows, head_dimension) ||
        tensor_init(&cache->merged, rows, block->d_model) ||
        tensor_init(&cache->attention_branch, rows, block->d_model) ||
        tensor_init(&cache->attention_residual, rows, block->d_model) ||
        tensor_init(&cache->normalized_mlp, rows, block->d_model) ||
        tensor_init(&cache->hidden, rows, 4 * block->d_model) ||
        tensor_init(&cache->activated, rows, 4 * block->d_model) ||
        tensor_init(&cache->mlp_branch, rows, block->d_model) ||
        tensor_init(&cache->output, rows, block->d_model);
    if (result != 0) {
        free_block_cache_tensors(cache);
        return -1;
    }
    return 0;
}

void vit_encoder_block_cache_free(ViTEncoderBlockCache *cache) {
    if (cache) {
        free_block_cache_tensors(cache);
        memset(cache, 0, sizeof(*cache));
    }
}

int vit_encoder_block_forward_cached(const ViTEncoderBlock *block,
                                     const Tensor *input,
                                     ViTEncoderBlockCache *cache) {
    if (!block || !input || !cache || cache->batch == 0 ||
        input->ndim != 2 ||
        input->rows != cache->batch * block->sequence_length ||
        input->cols != block->d_model) {
        return -1;
    }
    const size_t head_dimension = block->d_model / block->heads;
    int result = ops_layer_norm(input, &block->attention_gamma.value,
                                &block->attention_beta.value, 1e-5f,
                                &cache->normalized_attention) ||
        ops_gemm(&cache->normalized_attention, &block->query_key_value.value,
                 &cache->projected) ||
        ops_bias_add(&cache->projected, &block->query_key_value_bias.value,
                     &cache->projected);
    if (result == 0) {
        result = ops_split_heads(&cache->projected, &cache->query,
                                 &cache->key, &cache->value, cache->batch,
                                 block->sequence_length, block->d_model,
                                 block->heads) ||
            ops_multi_head_attention(
            &cache->query, &cache->key, &cache->value, cache->batch,
            block->heads, block->sequence_length,
            1.0f / sqrtf((float)head_dimension), &cache->probabilities,
            &cache->attended);
    }
    if (result == 0) {
        result = ops_merge_heads(&cache->attended, &cache->merged,
                                 cache->batch, block->sequence_length,
                                 block->d_model, block->heads) ||
            ops_gemm(&cache->merged, &block->attention_output.value,
                          &cache->attention_branch) ||
            ops_bias_add(&cache->attention_branch,
                         &block->attention_output_bias.value,
                         &cache->attention_branch) ||
            ops_residual(input, &cache->attention_branch,
                         &cache->attention_residual) ||
            ops_layer_norm(&cache->attention_residual, &block->mlp_gamma.value,
                           &block->mlp_beta.value, 1e-5f,
                           &cache->normalized_mlp) ||
            ops_gemm(&cache->normalized_mlp, &block->mlp_input.value,
                     &cache->hidden) ||
            ops_bias_add(&cache->hidden, &block->mlp_input_bias.value,
                         &cache->hidden) ||
            ops_gelu(&cache->hidden, &cache->activated) ||
            ops_gemm(&cache->activated, &block->mlp_output.value,
                     &cache->mlp_branch) ||
            ops_bias_add(&cache->mlp_branch, &block->mlp_output_bias.value,
                         &cache->mlp_branch) ||
            ops_residual(&cache->attention_residual, &cache->mlp_branch,
                         &cache->output);
    }
    return result;
}

int vit_encoder_block_backward(ViTEncoderBlock *block, const Tensor *input,
                               ViTEncoderBlockCache *cache,
                               Tensor *input_grad) {
    if (!block || !input || !cache || !input_grad ||
        input->rows != cache->output.rows || input->cols != block->d_model ||
        input_grad->rows != input->rows || input_grad->cols != input->cols) {
        return -1;
    }
    int result = ops_residual_backward(&cache->attention_residual,
                                       &cache->mlp_branch, &cache->output) ||
        ops_bias_add_backward(&cache->mlp_branch,
                              &block->mlp_output_bias.value) ||
        ops_gemm_backward(&cache->activated, &block->mlp_output.value,
                          &cache->mlp_branch) ||
        ops_gelu_backward(&cache->hidden, &cache->activated) ||
        ops_bias_add_backward(&cache->hidden, &block->mlp_input_bias.value) ||
        ops_gemm_backward(&cache->normalized_mlp, &block->mlp_input.value,
                          &cache->hidden) ||
        ops_layer_norm_backward(&cache->attention_residual,
                                &block->mlp_gamma.value, 1e-5f,
                                &cache->normalized_mlp,
                                &cache->attention_residual,
                                &block->mlp_gamma.value,
                                &block->mlp_beta.value);
    if (result != 0) {
        return result;
    }
    result = ops_accumulate_gradient(input_grad, &cache->attention_residual) ||
        ops_accumulate_gradient(&cache->attention_branch,
                                &cache->attention_residual) ||
        ops_bias_add_backward(&cache->attention_branch,
                                   &block->attention_output_bias.value) ||
        ops_gemm_backward(&cache->merged, &block->attention_output.value,
                          &cache->attention_branch);
    if (result != 0) {
        return result;
    }
    result = ops_merge_heads_backward(&cache->attended, &cache->merged,
                                      cache->batch, block->sequence_length,
                                      block->d_model, block->heads) ||
        ops_multi_head_attention_backward(
            &cache->query, &cache->key, &cache->value, cache->batch,
            block->heads, block->sequence_length,
            1.0f / sqrtf((float)(block->d_model / block->heads)),
            &cache->probabilities, &cache->attended, &cache->query,
            &cache->key, &cache->value) ||
        ops_split_heads_backward(&cache->projected, &cache->query, &cache->key,
                                 &cache->value, cache->batch,
                                 block->sequence_length, block->d_model,
                                 block->heads) ||
        ops_bias_add_backward(&cache->projected,
                              &block->query_key_value_bias.value) ||
        ops_gemm_backward(&cache->normalized_attention,
                          &block->query_key_value.value, &cache->projected) ||
        ops_layer_norm_backward(input, &block->attention_gamma.value, 1e-5f,
                                &cache->normalized_attention, input_grad,
                                &block->attention_gamma.value,
                                &block->attention_beta.value);
    return result;
}

int vit_classification_head_init(ViTClassificationHead *head, size_t d_model,
                                 size_t classes) {
    if (!head || d_model == 0 || classes == 0) {
        return -1;
    }
    memset(head, 0, sizeof(*head));
    head->d_model = d_model;
    head->classes = classes;
    if (parameter_init(&head->projection, d_model, classes) != 0 ||
        parameter_init(&head->bias, 1, classes) != 0) {
        vit_classification_head_free(head);
        return -1;
    }
    return 0;
}

void vit_classification_head_free(ViTClassificationHead *head) {
    if (!head) {
        return;
    }
    parameter_free(&head->projection);
    parameter_free(&head->bias);
    memset(head, 0, sizeof(*head));
}

int vit_classification_head_forward(const ViTClassificationHead *head,
                                    const Tensor *tokens, size_t batch,
                                    size_t sequence_length, Tensor *logits) {
    ops_synchronize();
    if (!head || !tokens || !logits || batch == 0 || sequence_length == 0 ||
        tokens->ndim != 2 || logits->ndim != 2 ||
        tokens->rows != batch * sequence_length ||
        tokens->cols != head->d_model || logits->rows != batch ||
        logits->cols != head->classes) {
        return -1;
    }
    for (size_t sample = 0; sample < batch; ++sample) {
        const float *cls_token = tokens->data +
                                 sample * sequence_length * head->d_model;
        for (size_t class_index = 0; class_index < head->classes; ++class_index) {
            float value = head->bias.value.data[class_index];
            for (size_t dimension = 0; dimension < head->d_model; ++dimension) {
                value += cls_token[dimension] *
                         head->projection.value.data[
                             dimension * head->classes + class_index];
            }
            logits->data[sample * logits->cols + class_index] = value;
        }
    }
    return 0;
}

int vit_classification_head_backward(ViTClassificationHead *head,
                                     const Tensor *tokens, size_t batch,
                                     size_t sequence_length,
                                     const Tensor *logits, Tensor *token_grad) {
    ops_synchronize();
    if (!head || !tokens || !logits || !token_grad || batch == 0 ||
        sequence_length == 0 || tokens->ndim != 2 || logits->ndim != 2 ||
        token_grad->ndim != 2 || tokens->rows != batch * sequence_length ||
        tokens->cols != head->d_model || logits->rows != batch ||
        logits->cols != head->classes || token_grad->rows != tokens->rows ||
        token_grad->cols != tokens->cols) {
        return -1;
    }
    for (size_t sample = 0; sample < batch; ++sample) {
        const float *cls_token = tokens->data +
                                 sample * sequence_length * head->d_model;
        float *cls_gradient = token_grad->grad +
                              sample * sequence_length * head->d_model;
        for (size_t class_index = 0; class_index < head->classes; ++class_index) {
            const float upstream =
                logits->grad[sample * logits->cols + class_index];
            head->bias.value.grad[class_index] += upstream;
            for (size_t dimension = 0; dimension < head->d_model; ++dimension) {
                const size_t weight_index = dimension * head->classes + class_index;
                head->projection.value.grad[weight_index] +=
                    cls_token[dimension] * upstream;
                cls_gradient[dimension] +=
                    head->projection.value.data[weight_index] * upstream;
            }
        }
    }
    return 0;
}

int vit_model_init_seeded(ViTModel *model, const ViTConfig *config,
                          uint64_t seed) {
    if (!model || !config || config->channels == 0 || config->height == 0 ||
        config->width == 0 || config->patch_size == 0 ||
        config->d_model == 0 || config->heads == 0 || config->layers == 0 ||
        config->classes == 0 || config->height % config->patch_size != 0 ||
        config->width % config->patch_size != 0 ||
        config->d_model % config->heads != 0) {
        return -1;
    }
    memset(model, 0, sizeof(*model));
    model->config = *config;
    model->patch_count = (config->height / config->patch_size) *
                         (config->width / config->patch_size);
    model->sequence_length = model->patch_count + 1;
    if (vit_patch_projection_init(&model->patch_projection, config->channels,
                                  config->height, config->width,
                                  config->patch_size, config->d_model) != 0 ||
        vit_token_embedding_init(&model->token_embedding, model->patch_count,
                                 config->d_model) != 0 ||
        parameter_init(&model->final_gamma, 1, config->d_model) != 0 ||
        parameter_init(&model->final_beta, 1, config->d_model) != 0 ||
        vit_classification_head_init(&model->classification_head,
                                     config->d_model, config->classes) != 0) {
        vit_model_free(model);
        return -1;
    }
    model->blocks = calloc(config->layers, sizeof(*model->blocks));
    if (!model->blocks) {
        vit_model_free(model);
        return -1;
    }
    for (size_t layer = 0; layer < config->layers; ++layer) {
        if (vit_encoder_block_init(&model->blocks[layer], config->d_model,
                                   config->heads,
                                   model->sequence_length) != 0) {
            vit_model_free(model);
            return -1;
        }
    }
    for (size_t dimension = 0; dimension < config->d_model; ++dimension) {
        model->final_gamma.value.data[dimension] = 1.0f;
    }
    const float embedding_standard_deviation = 0.02f;
    uint64_t random_state = seed;
    initialize_xavier_uniform(&model->patch_projection.projection,
                              &random_state);
    initialize_truncated_normal(&model->token_embedding.cls_token,
                                embedding_standard_deviation, &random_state);
    initialize_truncated_normal(&model->token_embedding.positional_embeddings,
                                embedding_standard_deviation, &random_state);
    for (size_t layer = 0; layer < config->layers; ++layer) {
        ViTEncoderBlock *block = &model->blocks[layer];
        initialize_xavier_uniform(&block->query_key_value, &random_state);
        initialize_xavier_uniform(&block->attention_output, &random_state);
        initialize_xavier_uniform(&block->mlp_input, &random_state);
        initialize_xavier_uniform(&block->mlp_output, &random_state);
    }
    initialize_truncated_normal(&model->classification_head.projection,
                                embedding_standard_deviation, &random_state);
    return 0;
}

int vit_model_init(ViTModel *model, const ViTConfig *config) {
    return vit_model_init_seeded(model, config, 42);
}

void vit_model_free(ViTModel *model) {
    if (!model) {
        return;
    }
    if (model->blocks) {
        for (size_t layer = 0; layer < model->config.layers; ++layer) {
            vit_encoder_block_free(&model->blocks[layer]);
        }
        free(model->blocks);
    }
    vit_patch_projection_free(&model->patch_projection);
    vit_token_embedding_free(&model->token_embedding);
    parameter_free(&model->final_gamma);
    parameter_free(&model->final_beta);
    vit_classification_head_free(&model->classification_head);
    memset(model, 0, sizeof(*model));
}

typedef struct {
    Parameter *parameter;
    int uses_weight_decay;
} TrainableParameter;

static size_t trainable_parameter_count(const ViTModel *model) {
    const size_t embedding_and_head_parameters = 8;
    const size_t parameters_per_block = 12;
    return embedding_and_head_parameters +
           parameters_per_block * model->config.layers;
}

static TrainableParameter *collect_trainable_parameters(ViTModel *model) {
    TrainableParameter *parameters =
        malloc(trainable_parameter_count(model) * sizeof(*parameters));
    if (!parameters) {
        return NULL;
    }
    size_t count = 0;
#define ADD_PARAMETER(target, decays) \
    parameters[count++] = (TrainableParameter){&(target), (decays)}
    ADD_PARAMETER(model->patch_projection.projection, 1);
    ADD_PARAMETER(model->patch_projection.bias, 0);
    ADD_PARAMETER(model->token_embedding.cls_token, 0);
    ADD_PARAMETER(model->token_embedding.positional_embeddings, 0);
    for (size_t layer = 0; layer < model->config.layers; ++layer) {
        ViTEncoderBlock *block = &model->blocks[layer];
        ADD_PARAMETER(block->query_key_value, 1);
        ADD_PARAMETER(block->query_key_value_bias, 0);
        ADD_PARAMETER(block->attention_output, 1);
        ADD_PARAMETER(block->attention_output_bias, 0);
        ADD_PARAMETER(block->mlp_input, 1);
        ADD_PARAMETER(block->mlp_input_bias, 0);
        ADD_PARAMETER(block->mlp_output, 1);
        ADD_PARAMETER(block->mlp_output_bias, 0);
        ADD_PARAMETER(block->attention_gamma, 0);
        ADD_PARAMETER(block->attention_beta, 0);
        ADD_PARAMETER(block->mlp_gamma, 0);
        ADD_PARAMETER(block->mlp_beta, 0);
    }
    ADD_PARAMETER(model->final_gamma, 0);
    ADD_PARAMETER(model->final_beta, 0);
    ADD_PARAMETER(model->classification_head.projection, 1);
    ADD_PARAMETER(model->classification_head.bias, 0);
#undef ADD_PARAMETER
    return parameters;
}

void vit_model_zero_grad(ViTModel *model) {
    if (!model) {
        return;
    }
    const size_t count = trainable_parameter_count(model);
    TrainableParameter *parameters = collect_trainable_parameters(model);
    Tensor **values = malloc(count * sizeof(*values));
    if (parameters && values) {
        for (size_t i = 0; i < count; ++i) {
            values[i] = &parameters[i].parameter->value;
        }
        ops_zero_gradients(values, count);
    }
    free(values);
    free(parameters);
}

int vit_model_cache_init(ViTModelCache *cache, const ViTModel *model,
                         size_t batch) {
    if (!cache || !model || batch == 0) {
        return -1;
    }
    memset(cache, 0, sizeof(*cache));
    cache->batch = batch;
    cache->layers = model->config.layers;
    const ViTConfig *config = &model->config;
    const size_t token_rows = batch * model->sequence_length;
    const size_t image_size = config->channels * config->height * config->width;
    const size_t patch_elements =
        config->channels * config->patch_size * config->patch_size;
    if (tensor_init(&cache->images, batch, image_size) != 0 ||
        tensor_init(&cache->patches, batch * model->patch_count,
                    patch_elements) != 0 ||
        tensor_init(&cache->patch_tokens, batch * model->patch_count,
                    config->d_model) != 0 ||
        tensor_init(&cache->tokens, token_rows, config->d_model) != 0 ||
        tensor_init(&cache->normalized_tokens, token_rows,
                    config->d_model) != 0 ||
        tensor_init(&cache->cls_tokens, batch, config->d_model) != 0 ||
        tensor_init(&cache->logits, batch, config->classes) != 0 ||
        tensor_init(&cache->targets, batch, 1) != 0 ||
        tensor_init(&cache->class_weights, 1, config->classes) != 0 ||
        tensor_init(&cache->loss, 1, 1) != 0) {
        vit_model_cache_free(cache);
        return -1;
    }
    cache->block_caches = calloc(config->layers, sizeof(*cache->block_caches));
    if (!cache->block_caches) {
        vit_model_cache_free(cache);
        return -1;
    }
    for (size_t layer = 0; layer < config->layers; ++layer) {
        if (vit_encoder_block_cache_init(&cache->block_caches[layer],
                                         &model->blocks[layer], batch) != 0) {
            vit_model_cache_free(cache);
            return -1;
        }
    }
    return 0;
}

void vit_model_cache_free(ViTModelCache *cache) {
    if (!cache) {
        return;
    }
    ops_synchronize();
    if (cache->block_caches) {
        for (size_t layer = 0; layer < cache->layers; ++layer) {
            vit_encoder_block_cache_free(&cache->block_caches[layer]);
        }
        free(cache->block_caches);
    }
    tensor_free(&cache->images);
    tensor_free(&cache->patches);
    tensor_free(&cache->patch_tokens);
    tensor_free(&cache->tokens);
    tensor_free(&cache->normalized_tokens);
    tensor_free(&cache->cls_tokens);
    tensor_free(&cache->logits);
    tensor_free(&cache->targets);
    tensor_free(&cache->class_weights);
    tensor_free(&cache->loss);
    memset(cache, 0, sizeof(*cache));
}

int vit_model_clip_gradients(ViTModel *model, float max_norm) {
    if (!model || max_norm <= 0.0f) {
        return -1;
    }
    const size_t count = trainable_parameter_count(model);
    TrainableParameter *trainable = collect_trainable_parameters(model);
    Parameter **parameters = malloc(count * sizeof(*parameters));
    int result = -1;
    if (trainable && parameters) {
        for (size_t i = 0; i < count; ++i) {
            parameters[i] = trainable[i].parameter;
        }
        result = ops_clip_gradient_norm(parameters, count, max_norm);
    }
    free(parameters);
    free(trainable);
    return result;
}

static int vit_model_encode_forward(const ViTModel *model, const float *images,
                                    ViTModelCache *cache) {
    const ViTConfig *config = &model->config;
    int result = ops_upload_values(&cache->images, images) ||
        ops_extract_patches(&cache->images, &cache->patches, config->channels,
                            config->height, config->width,
                            config->patch_size) ||
        ops_gemm(&cache->patches, &model->patch_projection.projection.value,
                 &cache->patch_tokens) ||
        ops_bias_add(&cache->patch_tokens, &model->patch_projection.bias.value,
                     &cache->patch_tokens) ||
        ops_token_embedding(&cache->patch_tokens,
                            &model->token_embedding.cls_token.value,
                            &model->token_embedding.positional_embeddings.value,
                            &cache->tokens, cache->batch);
    const Tensor *block_input = &cache->tokens;
    for (size_t layer = 0; result == 0 && layer < config->layers; ++layer) {
        result = vit_encoder_block_forward_cached(&model->blocks[layer],
                                                  block_input,
                                                  &cache->block_caches[layer]);
        block_input = &cache->block_caches[layer].output;
    }
    return result ||
        ops_layer_norm(block_input, &model->final_gamma.value,
                       &model->final_beta.value, 1e-5f,
                       &cache->normalized_tokens) ||
        ops_gather_rows(&cache->normalized_tokens, &cache->cls_tokens,
                        model->sequence_length) ||
        ops_gemm(&cache->cls_tokens,
                 &model->classification_head.projection.value,
                 &cache->logits) ||
        ops_bias_add(&cache->logits, &model->classification_head.bias.value,
                     &cache->logits);
}

int vit_model_forward(const ViTModel *model, const float *images, size_t batch,
                      ViTModelCache *cache) {
    if (!model || !images || !cache || batch == 0 || cache->batch != batch) {
        return -1;
    }
    const int result = vit_model_encode_forward(model, images, cache);
    ops_synchronize();
    return result;
}

int vit_model_backward(ViTModel *model, const float *images,
                       ViTModelCache *cache, float *image_grad) {
    if (!model || !images || !cache || cache->batch == 0) {
        return -1;
    }
    const ViTConfig *config = &model->config;
    Tensor *last_block_output = &cache->block_caches[config->layers - 1].output;
    int result = ops_bias_add_backward(&cache->logits,
                                       &model->classification_head.bias.value) ||
        ops_gemm_backward(&cache->cls_tokens,
                          &model->classification_head.projection.value,
                          &cache->logits) ||
        ops_gather_rows_backward(&cache->normalized_tokens, &cache->cls_tokens,
                                 model->sequence_length) ||
        ops_layer_norm_backward(last_block_output, &model->final_gamma.value,
                                1e-5f, &cache->normalized_tokens,
                                last_block_output, &model->final_gamma.value,
                                &model->final_beta.value);
    for (size_t layer = config->layers; result == 0 && layer > 0; --layer) {
        const size_t index = layer - 1;
        Tensor *block_input = index == 0
            ? &cache->tokens
            : &cache->block_caches[index - 1].output;
        result = vit_encoder_block_backward(&model->blocks[index], block_input,
                                            &cache->block_caches[index],
                                            block_input);
    }
    result = result ||
        ops_token_embedding_backward(
            &cache->patch_tokens, &model->token_embedding.cls_token.value,
            &model->token_embedding.positional_embeddings.value,
            &cache->tokens, cache->batch) ||
        ops_bias_add_backward(&cache->patch_tokens,
                              &model->patch_projection.bias.value) ||
        ops_gemm_backward(&cache->patches,
                          &model->patch_projection.projection.value,
                          &cache->patch_tokens);
    if (result == 0 && image_grad) {
        result = ops_extract_patches_backward(&cache->patches, image_grad,
                                              config->channels, config->height,
                                              config->width,
                                              config->patch_size);
    }
    return result;
}

static int vit_model_zero_cache_grad(ViTModelCache *cache) {
    Tensor *model_tensors[] = {
        &cache->images, &cache->patches, &cache->patch_tokens, &cache->tokens,
        &cache->normalized_tokens, &cache->cls_tokens, &cache->logits,
        &cache->targets, &cache->class_weights, &cache->loss,
    };
    const size_t model_tensor_count =
        sizeof(model_tensors) / sizeof(*model_tensors);
    const size_t tensors_per_block = 15;
    const size_t count = model_tensor_count + tensors_per_block * cache->layers;
    Tensor **tensors = malloc(count * sizeof(*tensors));
    if (!tensors) {
        return -1;
    }
    size_t index = 0;
    for (size_t i = 0; i < model_tensor_count; ++i) {
        tensors[index++] = model_tensors[i];
    }
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        ViTEncoderBlockCache *block = &cache->block_caches[layer];
        Tensor *block_tensors[] = {
            &block->normalized_attention, &block->projected, &block->query,
            &block->key, &block->value, &block->probabilities,
            &block->attended, &block->merged, &block->attention_branch,
            &block->attention_residual, &block->normalized_mlp,
            &block->hidden, &block->activated, &block->mlp_branch,
            &block->output,
        };
        for (size_t i = 0; i < tensors_per_block; ++i) {
            tensors[index++] = block_tensors[i];
        }
    }
    const int result = ops_zero_gradients(tensors, count);
    free(tensors);
    return result;
}

int vit_model_train_batch(ViTModel *model, const float *images,
                          const size_t *targets, size_t batch,
                          ViTModelCache *cache, float learning_rate,
                          float weight_decay, float *loss) {
    return vit_model_train_batch_weighted(
        model, images, targets, batch, cache, learning_rate, weight_decay,
        NULL, loss);
}

int vit_model_train_batch_weighted(
    ViTModel *model, const float *images, const size_t *targets, size_t batch,
    ViTModelCache *cache, float learning_rate, float weight_decay,
    const float *class_weights, float *loss) {
    return vit_model_train_batch_smoothed(
        model, images, targets, batch, cache, learning_rate, weight_decay,
        class_weights, 0.0f, loss);
}

static int vit_model_apply_clipped_adamw(ViTModel *model, float learning_rate,
                                         float weight_decay,
                                         float max_gradient_norm) {
    const size_t count = trainable_parameter_count(model);
    TrainableParameter *trainable = collect_trainable_parameters(model);
    Parameter **parameters = malloc(count * sizeof(*parameters));
    float *weight_decays = malloc(count * sizeof(*weight_decays));
    int result = -1;
    if (trainable && parameters && weight_decays) {
        for (size_t i = 0; i < count; ++i) {
            parameters[i] = trainable[i].parameter;
            weight_decays[i] = trainable[i].uses_weight_decay ? weight_decay
                                                              : 0.0f;
        }
        result = ops_adamw_step_parameters(parameters, weight_decays, count,
                                           learning_rate, 0.9f, 0.999f, 1e-8f,
                                           max_gradient_norm);
    }
    free(weight_decays);
    free(parameters);
    free(trainable);
    return result;
}

int vit_model_train_batch_smoothed(
    ViTModel *model, const float *images, const size_t *targets, size_t batch,
    ViTModelCache *cache, float learning_rate, float weight_decay,
    const float *class_weights, float label_smoothing, float *loss) {
    if (!model || !images || !targets || !cache || !loss || batch == 0 ||
        cache->batch != batch || learning_rate <= 0.0f || weight_decay < 0.0f) {
        return -1;
    }
    const float max_gradient_norm = 1.0f;
    vit_model_zero_grad(model);
    int result = vit_model_zero_cache_grad(cache) ||
        ops_upload_labels(&cache->targets, targets) ||
        (class_weights &&
         ops_upload_values(&cache->class_weights, class_weights)) ||
        vit_model_encode_forward(model, images, cache) ||
        ops_softmax_cross_entropy_tensor(
            &cache->logits, &cache->targets,
            class_weights ? &cache->class_weights : NULL, label_smoothing,
            &cache->loss, &cache->logits) ||
        vit_model_backward(model, images, cache, NULL) ||
        vit_model_apply_clipped_adamw(model, learning_rate, weight_decay,
                                      max_gradient_norm);
    ops_synchronize();
    if (result == 0) {
        *loss = cache->loss.data[0];
    }
    return result;
}

static int write_parameter(FILE *file, const Parameter *parameter) {
    const size_t count = tensor_numel(&parameter->value);
    return fwrite(&count, sizeof(count), 1, file) == 1 &&
        fwrite(&parameter->step, sizeof(parameter->step), 1, file) == 1 &&
        fwrite(parameter->value.data, sizeof(float), count, file) == count &&
        fwrite(parameter->m, sizeof(float), count, file) == count &&
        fwrite(parameter->v, sizeof(float), count, file) == count;
}

static int read_parameter(FILE *file, Parameter *parameter) {
    size_t count = 0;
    if (fread(&count, sizeof(count), 1, file) != 1 ||
        count != tensor_numel(&parameter->value) ||
        fread(&parameter->step, sizeof(parameter->step), 1, file) != 1 ||
        fread(parameter->value.data, sizeof(float), count, file) != count ||
        fread(parameter->m, sizeof(float), count, file) != count ||
        fread(parameter->v, sizeof(float), count, file) != count) {
        return -1;
    }
    return 0;
}

static int write_model_parameters(FILE *file, const ViTModel *model) {
    if (!write_parameter(file, &model->patch_projection.projection) ||
        !write_parameter(file, &model->patch_projection.bias) ||
        !write_parameter(file, &model->token_embedding.cls_token) ||
        !write_parameter(file, &model->token_embedding.positional_embeddings)) {
        return -1;
    }
    for (size_t layer = 0; layer < model->config.layers; ++layer) {
        const ViTEncoderBlock *block = &model->blocks[layer];
        if (!write_parameter(file, &block->query_key_value) ||
            !write_parameter(file, &block->query_key_value_bias) ||
            !write_parameter(file, &block->attention_output) ||
            !write_parameter(file, &block->attention_output_bias) ||
            !write_parameter(file, &block->mlp_input) ||
            !write_parameter(file, &block->mlp_input_bias) ||
            !write_parameter(file, &block->mlp_output) ||
            !write_parameter(file, &block->mlp_output_bias) ||
            !write_parameter(file, &block->attention_gamma) ||
            !write_parameter(file, &block->attention_beta) ||
            !write_parameter(file, &block->mlp_gamma) ||
            !write_parameter(file, &block->mlp_beta)) {
            return -1;
        }
    }
    return write_parameter(file, &model->final_gamma) &&
        write_parameter(file, &model->final_beta) &&
        write_parameter(file, &model->classification_head.projection) &&
        write_parameter(file, &model->classification_head.bias) ? 0 : -1;
}

static int read_model_parameters(FILE *file, ViTModel *model) {
    if (read_parameter(file, &model->patch_projection.projection) != 0 ||
        read_parameter(file, &model->patch_projection.bias) != 0 ||
        read_parameter(file, &model->token_embedding.cls_token) != 0 ||
        read_parameter(file, &model->token_embedding.positional_embeddings) != 0) {
        return -1;
    }
    for (size_t layer = 0; layer < model->config.layers; ++layer) {
        ViTEncoderBlock *block = &model->blocks[layer];
        if (read_parameter(file, &block->query_key_value) != 0 ||
            read_parameter(file, &block->query_key_value_bias) != 0 ||
            read_parameter(file, &block->attention_output) != 0 ||
            read_parameter(file, &block->attention_output_bias) != 0 ||
            read_parameter(file, &block->mlp_input) != 0 ||
            read_parameter(file, &block->mlp_input_bias) != 0 ||
            read_parameter(file, &block->mlp_output) != 0 ||
            read_parameter(file, &block->mlp_output_bias) != 0 ||
            read_parameter(file, &block->attention_gamma) != 0 ||
            read_parameter(file, &block->attention_beta) != 0 ||
            read_parameter(file, &block->mlp_gamma) != 0 ||
            read_parameter(file, &block->mlp_beta) != 0) {
            return -1;
        }
    }
    return read_parameter(file, &model->final_gamma) ||
        read_parameter(file, &model->final_beta) ||
        read_parameter(file, &model->classification_head.projection) ||
        read_parameter(file, &model->classification_head.bias);
}

int vit_model_save(const ViTModel *model, const char *path) {
    ops_synchronize();
    static const char magic[8] = {'C', 'V', 'I', 'T', 'C', 'K', 'P', '\0'};
    const uint32_t version = 1;
    if (!model || !path) {
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        return -1;
    }
    int result = fwrite(magic, sizeof(magic), 1, file) == 1 &&
        fwrite(&version, sizeof(version), 1, file) == 1 &&
        fwrite(&model->config, sizeof(model->config), 1, file) == 1 &&
        write_model_parameters(file, model) == 0;
    if (fclose(file) != 0) {
        result = 0;
    }
    return result ? 0 : -1;
}

static int read_checkpoint_header(FILE *file, ViTConfig *config) {
    static const char magic[8] = {'C', 'V', 'I', 'T', 'C', 'K', 'P', '\0'};
    const uint32_t supported_version = 1;
    char file_magic[sizeof(magic)];
    uint32_t version = 0;
    return fread(file_magic, sizeof(file_magic), 1, file) == 1 &&
           memcmp(file_magic, magic, sizeof(magic)) == 0 &&
           fread(&version, sizeof(version), 1, file) == 1 &&
           version == supported_version &&
           fread(config, sizeof(*config), 1, file) == 1
        ? 0
        : -1;
}

int vit_model_read_config(const char *path, ViTConfig *config) {
    if (!path || !config) {
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }
    const int result = read_checkpoint_header(file, config);
    fclose(file);
    return result;
}

int vit_model_load(ViTModel *model, const char *path) {
    ops_synchronize();
    ViTConfig config = {0};
    if (!model || !path) {
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }
    const int result = read_checkpoint_header(file, &config) == 0 &&
        memcmp(&config, &model->config, sizeof(config)) == 0 &&
        read_model_parameters(file, model) == 0;
    fclose(file);
    return result ? 0 : -1;
}

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static size_t count_correct_predictions(const Tensor *logits,
                                        const size_t *targets) {
    size_t correct = 0;
    for (size_t sample = 0; sample < logits->rows; ++sample) {
        const float *sample_logits = logits->data + sample * logits->cols;
        size_t prediction = 0;
        for (size_t class_index = 1; class_index < logits->cols; ++class_index) {
            if (sample_logits[class_index] > sample_logits[prediction]) {
                prediction = class_index;
            }
        }
        correct += prediction == targets[sample];
    }
    return correct;
}

static int evaluate_chunk(const ViTModel *model, const float *images,
                          const size_t *targets, ViTModelCache *cache,
                          double *loss_sum, size_t *correct) {
    float chunk_loss = 0.0f;
    if (vit_model_forward(model, images, cache->batch, cache) != 0 ||
        ops_softmax_cross_entropy(&cache->logits, targets, &chunk_loss,
                                  &cache->logits) != 0) {
        return -1;
    }
    *loss_sum += (double)chunk_loss * (double)cache->batch;
    *correct += count_correct_predictions(&cache->logits, targets);
    return 0;
}

static void softmax_rows(const Tensor *logits, float *probabilities) {
    for (size_t row = 0; row < logits->rows; ++row) {
        const float *row_logits = logits->data + row * logits->cols;
        float *row_probabilities = probabilities + row * logits->cols;
        float maximum = row_logits[0];
        for (size_t col = 1; col < logits->cols; ++col) {
            maximum = fmaxf(maximum, row_logits[col]);
        }
        float denominator = 0.0f;
        for (size_t col = 0; col < logits->cols; ++col) {
            row_probabilities[col] = expf(row_logits[col] - maximum);
            denominator += row_probabilities[col];
        }
        for (size_t col = 0; col < logits->cols; ++col) {
            row_probabilities[col] /= denominator;
        }
    }
}

int vit_model_predict_probabilities(const ViTModel *model, const float *images,
                                    size_t sample_count, float *probabilities) {
    if (!model || !images || !probabilities || sample_count == 0) {
        return -1;
    }
    const size_t maximum_chunk_size = 256;
    const size_t image_size = model->config.channels * model->config.height *
                              model->config.width;
    ViTModelCache cache = {0};
    int result = 0;
    for (size_t offset = 0; result == 0 && offset < sample_count;
         offset += maximum_chunk_size) {
        const size_t remaining = sample_count - offset;
        const size_t chunk_size = remaining < maximum_chunk_size
            ? remaining
            : maximum_chunk_size;
        if (chunk_size != cache.batch) {
            vit_model_cache_free(&cache);
            result = vit_model_cache_init(&cache, model, chunk_size);
        }
        result = result ||
            vit_model_forward(model, images + offset * image_size, chunk_size,
                              &cache);
        if (result == 0) {
            softmax_rows(&cache.logits,
                         probabilities + offset * model->config.classes);
        }
    }
    vit_model_cache_free(&cache);
    return result;
}

int vit_model_evaluate(const ViTModel *model, const float *images,
                       const size_t *targets, size_t sample_count,
                       ViTEvaluationMetrics *metrics) {
    if (!model || !images || !targets || !metrics || sample_count == 0) {
        return -1;
    }
    const size_t maximum_chunk_size = 256;
    const size_t chunk_size = sample_count < maximum_chunk_size
        ? sample_count
        : maximum_chunk_size;
    const size_t full_chunks = sample_count / chunk_size;
    const size_t remainder = sample_count - full_chunks * chunk_size;
    const size_t image_size = model->config.channels * model->config.height *
                              model->config.width;
    ViTModelCache chunk_cache = {0};
    ViTModelCache remainder_cache = {0};
    int result = vit_model_cache_init(&chunk_cache, model, chunk_size) ||
        (remainder > 0 &&
         vit_model_cache_init(&remainder_cache, model, remainder));
    double loss_sum = 0.0;
    size_t correct = 0;
    const double start = monotonic_seconds();
    for (size_t chunk = 0; result == 0 && chunk < full_chunks; ++chunk) {
        const size_t offset = chunk * chunk_size;
        result = evaluate_chunk(model, images + offset * image_size,
                                targets + offset, &chunk_cache, &loss_sum,
                                &correct);
    }
    if (result == 0 && remainder > 0) {
        const size_t offset = full_chunks * chunk_size;
        result = evaluate_chunk(model, images + offset * image_size,
                                targets + offset, &remainder_cache, &loss_sum,
                                &correct);
    }
    if (result == 0) {
        metrics->accuracy = (float)correct / (float)sample_count;
        metrics->average_loss = (float)(loss_sum / (double)sample_count);
        metrics->elapsed_seconds = monotonic_seconds() - start;
        metrics->samples_per_second =
            metrics->elapsed_seconds > 0.0
                ? (double)sample_count / metrics->elapsed_seconds
                : 0.0;
    }
    vit_model_cache_free(&remainder_cache);
    vit_model_cache_free(&chunk_cache);
    return result;
}

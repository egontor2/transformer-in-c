#include "vit.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        initialize_parameter(&block->attention_output, d_model, d_model) ||
        initialize_parameter(&block->mlp_input, d_model, 4 * d_model) ||
        initialize_parameter(&block->mlp_output, 4 * d_model, d_model) ||
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
    parameter_free(&block->attention_output);
    parameter_free(&block->mlp_input);
    parameter_free(&block->mlp_output);
    parameter_free(&block->attention_gamma);
    parameter_free(&block->attention_beta);
    parameter_free(&block->mlp_gamma);
    parameter_free(&block->mlp_beta);
    memset(block, 0, sizeof(*block));
}

static int create_matrix(Tensor *tensor, size_t rows, size_t cols) {
    return tensor_init(tensor, rows, cols);
}

static void split_attention_inputs(const Tensor *projected, Tensor *query,
                                   Tensor *key, Tensor *value, size_t batch,
                                   size_t sequence_length, size_t d_model,
                                   size_t heads) {
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const size_t source_row = sample * sequence_length + position;
            for (size_t head = 0; head < heads; ++head) {
                const size_t target_row =
                    (sample * heads + head) * sequence_length + position;
                for (size_t dimension = 0; dimension < head_dimension; ++dimension) {
                    const size_t offset = head * head_dimension + dimension;
                    query->data[target_row * head_dimension + dimension] =
                        projected->data[source_row * (3 * d_model) + offset];
                    key->data[target_row * head_dimension + dimension] =
                        projected->data[source_row * (3 * d_model) + d_model + offset];
                    value->data[target_row * head_dimension + dimension] =
                        projected->data[source_row * (3 * d_model) + 2 * d_model + offset];
                }
            }
        }
    }
}

static void merge_attention_output(const Tensor *attention, Tensor *merged,
                                   size_t batch, size_t sequence_length,
                                   size_t d_model, size_t heads) {
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const size_t output_row = sample * sequence_length + position;
            for (size_t head = 0; head < heads; ++head) {
                const size_t source_row =
                    (sample * heads + head) * sequence_length + position;
                for (size_t dimension = 0; dimension < head_dimension; ++dimension) {
                    merged->data[output_row * d_model + head * head_dimension +
                                 dimension] =
                        attention->data[source_row * head_dimension + dimension];
                }
            }
        }
    }
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
        split_attention_inputs(&projected, &query, &key, &value, batch,
                               block->sequence_length, block->d_model,
                               block->heads);
        result = ops_multi_head_attention(
            &query, &key, &value, batch, block->heads,
            block->sequence_length, 1.0f / sqrtf((float)head_dimension),
            &probabilities, &attended);
    }
    if (result == 0) {
        merge_attention_output(&attended, &merged, batch,
                               block->sequence_length, block->d_model,
                               block->heads);
        result = ops_gemm(&merged, &block->attention_output.value,
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

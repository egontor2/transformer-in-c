#include "vit.h"

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

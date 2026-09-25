#include "transformer.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *gamma;
    float *beta;
    float *qkv;
    float *wo;
    float *w1;
    float *w2;
} TransformerLayer;

struct Transformer {
    TransformerConfig config;
    TransformerLayer *layers;
    float *patch_projection;
    float *classifier;
};

static size_t matrix_size(size_t rows, size_t cols) {
    return rows * cols;
}

static float *allocate_vector(size_t size) {
    return calloc(size, sizeof(float));
}

static float *allocate_matrix(size_t rows, size_t cols) {
    return calloc(matrix_size(rows, cols), sizeof(float));
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float random_weight(uint32_t *state, float scale) {
    const float unit = (float)(next_random(state) & 0xffffu) / 65535.0f;
    return (unit * 2.0f - 1.0f) * scale;
}

static void initialize_matrix(float *matrix, size_t count, float scale, uint32_t *seed) {
    for (size_t i = 0; i < count; ++i) {
        matrix[i] = random_weight(seed, scale);
    }
}

static void free_layer(TransformerLayer *layer) {
    free(layer->gamma);
    free(layer->beta);
    free(layer->qkv);
    free(layer->wo);
    free(layer->w1);
    free(layer->w2);
}

Transformer *transformer_create(TransformerConfig config, unsigned int seed) {
    if (config.d_model == 0 || config.n_heads == 0 || config.d_ff == 0 ||
        config.n_layers == 0 || config.d_model % config.n_heads != 0) {
        return NULL;
    }

    Transformer *model = calloc(1, sizeof(*model));
    if (model == NULL) {
        return NULL;
    }
    model->config = config;
    model->layers = calloc(config.n_layers, sizeof(*model->layers));
    if (model->layers == NULL) {
        free(model);
        return NULL;
    }

    uint32_t state = seed == 0 ? 1u : seed;
    const float projection_scale = 1.0f / sqrtf((float)config.d_model);
    const float ffn_scale = 1.0f / sqrtf((float)config.d_model);
    if (config.patch_size != 0 && config.input_channels != 0) {
        const size_t patch_elements = config.patch_size * config.patch_size *
                                      config.input_channels;
        model->patch_projection = allocate_matrix(config.d_model, patch_elements);
        if (model->patch_projection == NULL) {
            transformer_free(model);
            return NULL;
        }
        if (config.n_classes != 0) {
            model->classifier = allocate_matrix(config.n_classes, config.d_model);
            if (model->classifier == NULL) {
                transformer_free(model);
                return NULL;
            }
            initialize_matrix(model->classifier, config.n_classes * config.d_model,
                              1.0f / sqrtf((float)config.d_model), &state);
        }
        initialize_matrix(model->patch_projection,
                          config.d_model * patch_elements,
                          1.0f / sqrtf((float)patch_elements), &state);
    }

    for (size_t i = 0; i < config.n_layers; ++i) {
        TransformerLayer *layer = &model->layers[i];
        layer->gamma = allocate_vector(config.d_model);
        layer->beta = allocate_vector(config.d_model);
        layer->qkv = allocate_matrix(3 * config.d_model, config.d_model);
        layer->wo = allocate_matrix(config.d_model, config.d_model);
        layer->w1 = allocate_matrix(config.d_ff, config.d_model);
        layer->w2 = allocate_matrix(config.d_model, config.d_ff);
        if (!layer->gamma || !layer->beta || !layer->qkv || !layer->wo ||
            !layer->w1 || !layer->w2) {
            transformer_free(model);
            return NULL;
        }

        for (size_t j = 0; j < config.d_model; ++j) {
            layer->gamma[j] = 1.0f;
        }
        initialize_matrix(layer->qkv, 3 * config.d_model * config.d_model,
                          projection_scale, &state);
        initialize_matrix(layer->wo, config.d_model * config.d_model,
                          projection_scale, &state);
        initialize_matrix(layer->w1, config.d_ff * config.d_model,
                          ffn_scale, &state);
        initialize_matrix(layer->w2, config.d_model * config.d_ff,
                          ffn_scale, &state);
    }
    return model;
}

void transformer_free(Transformer *model) {
    if (model == NULL) {
        return;
    }
    for (size_t i = 0; i < model->config.n_layers; ++i) {
        free_layer(&model->layers[i]);
    }
    free(model->layers);
    free(model->patch_projection);
    free(model->classifier);
    free(model);
}

static void layer_norm(
    const float *input,
    float *output,
    const float *gamma,
    size_t length,
    size_t width
) {
    for (size_t token = 0; token < length; ++token) {
        const float *row = input + token * width;
        float *normalized = output + token * width;
        float mean = 0.0f;
        for (size_t i = 0; i < width; ++i) {
            mean += row[i];
        }
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t i = 0; i < width; ++i) {
            const float centered = row[i] - mean;
            variance += centered * centered;
        }
        variance /= (float)width;
        const float inverse_std = 1.0f / sqrtf(variance + 1e-5f);
        for (size_t i = 0; i < width; ++i) {
            normalized[i] = (row[i] - mean) * inverse_std * gamma[i];
        }
    }
}

static void matmul(const float *input, const float *weights, float *output,
                   size_t rows, size_t input_width, size_t output_width) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < output_width; ++column) {
            float value = 0.0f;
            for (size_t i = 0; i < input_width; ++i) {
                value += input[row * input_width + i] *
                         weights[column * input_width + i];
            }
            output[row * output_width + column] = value;
        }
    }
}

static int self_attention(const TransformerConfig *config,
                          const TransformerLayer *layer,
                          const float *normalized,
                          float *output,
                          size_t length) {
    const size_t width = config->d_model;
    const size_t head_width = width / config->n_heads;
    float *qkv = calloc(length * 3 * width, sizeof(float));
    float *context = calloc(length * width, sizeof(float));
    float *scores = calloc(length, sizeof(float));
    if (!qkv || !context || !scores) {
        free(qkv);
        free(context);
        free(scores);
        return -1;
    }

    matmul(normalized, layer->qkv, qkv, length, width, 3 * width);
    const float scale = 1.0f / sqrtf((float)head_width);
    for (size_t head = 0; head < config->n_heads; ++head) {
        const size_t offset = head * head_width;
        for (size_t query = 0; query < length; ++query) {
            float maximum = -INFINITY;
            for (size_t key = 0; key < length; ++key) {
                float score = 0.0f;
                for (size_t i = 0; i < head_width; ++i) {
                    score += qkv[query * 3 * width + offset + i] *
                             qkv[key * 3 * width + width + offset + i];
                }
                scores[key] = score * scale;
                if (scores[key] > maximum) {
                    maximum = scores[key];
                }
            }
            float denominator = 0.0f;
            for (size_t key = 0; key < length; ++key) {
                scores[key] = expf(scores[key] - maximum);
                denominator += scores[key];
            }
            for (size_t key = 0; key < length; ++key) {
                const float probability = scores[key] / denominator;
                for (size_t i = 0; i < head_width; ++i) {
                    context[query * width + offset + i] +=
                        probability * qkv[key * 3 * width + 2 * width + offset + i];
                }
            }
        }
    }
    matmul(context, layer->wo, output, length, width, width);
    free(qkv);
    free(context);
    free(scores);
    return 0;
}

int transformer_forward(const Transformer *model, const float *input,
                        size_t length, float *output) {
    if (!model || !input || !output || length == 0) {
        return -1;
    }
    const size_t width = model->config.d_model;
    const size_t elements = length * width;
    float *state = malloc(elements * sizeof(float));
    float *normalized = malloc(elements * sizeof(float));
    float *attention = malloc(elements * sizeof(float));
    float *ffn_input = malloc(elements * sizeof(float));
    float *ffn = malloc(length * model->config.d_ff * sizeof(float));
    if (!state || !normalized || !attention || !ffn_input || !ffn) {
        free(state);
        free(normalized);
        free(attention);
        free(ffn_input);
        free(ffn);
        return -1;
    }
    memcpy(state, input, elements * sizeof(float));

    for (size_t layer_index = 0; layer_index < model->config.n_layers; ++layer_index) {
        const TransformerLayer *layer = &model->layers[layer_index];
        layer_norm(state, normalized, layer->gamma, length, width);
        if (self_attention(&model->config, layer, normalized, attention, length) != 0) {
            free(state);
            free(normalized);
            free(attention);
            free(ffn_input);
            free(ffn);
            return -1;
        }
        for (size_t i = 0; i < elements; ++i) {
            ffn_input[i] = state[i] + attention[i];
        }
        layer_norm(ffn_input, normalized, layer->gamma, length, width);
        matmul(normalized, layer->w1, ffn, length, width, model->config.d_ff);
        for (size_t i = 0; i < length * model->config.d_ff; ++i) {
            ffn[i] = ffn[i] > 0.0f ? ffn[i] : 0.0f;
        }
        matmul(ffn, layer->w2, attention, length, model->config.d_ff, width);
        for (size_t i = 0; i < elements; ++i) {
            state[i] = ffn_input[i] + attention[i];
        }
    }
    memcpy(output, state, elements * sizeof(float));
    free(state);
    free(normalized);
    free(attention);
    free(ffn_input);
    free(ffn);
    return 0;
}

static void add_position_encoding(float *tokens, size_t length, size_t width) {
    for (size_t position = 0; position < length; ++position) {
        for (size_t dimension = 0; dimension < width; ++dimension) {
            const float exponent = (float)(dimension - (dimension % 2)) /
                                   (float)width;
            const float angle = (float)position / powf(10000.0f, exponent);
            tokens[position * width + dimension] +=
                dimension % 2 == 0 ? sinf(angle) : cosf(angle);
        }
    }
}

int transformer_forward_image(const Transformer *model, const float *image,
                              size_t height, size_t width, float *output) {
    if (!model || !image || !output || height == 0 || width == 0 ||
        model->config.patch_size == 0 || model->config.input_channels == 0 ||
        model->patch_projection == NULL ||
        height % model->config.patch_size != 0 ||
        width % model->config.patch_size != 0) {
        return -1;
    }

    const size_t patch_size = model->config.patch_size;
    const size_t channels = model->config.input_channels;
    const size_t patches_y = height / patch_size;
    const size_t patches_x = width / patch_size;
    const size_t patch_count = patches_y * patches_x;
    const size_t patch_elements = patch_size * patch_size * channels;
    float *tokens = calloc(patch_count * model->config.d_model, sizeof(float));
    float *patch = malloc(patch_elements * sizeof(float));
    if (!tokens || !patch) {
        free(tokens);
        free(patch);
        return -1;
    }

    size_t token = 0;
    for (size_t py = 0; py < patches_y; ++py) {
        for (size_t px = 0; px < patches_x; ++px) {
            size_t patch_index = 0;
            for (size_t y = 0; y < patch_size; ++y) {
                for (size_t x = 0; x < patch_size; ++x) {
                    const size_t pixel = ((py * patch_size + y) * width +
                                          px * patch_size + x) * channels;
                    for (size_t channel = 0; channel < channels; ++channel) {
                        patch[patch_index++] = image[pixel + channel];
                    }
                }
            }
            for (size_t dimension = 0; dimension < model->config.d_model; ++dimension) {
                float value = 0.0f;
                for (size_t i = 0; i < patch_elements; ++i) {
                    value += patch[i] *
                             model->patch_projection[dimension * patch_elements + i];
                }
                tokens[token * model->config.d_model + dimension] = value;
            }
            ++token;
        }
    }
    add_position_encoding(tokens, patch_count, model->config.d_model);
    const int result = transformer_forward(model, tokens, patch_count, output);
    free(tokens);
    free(patch);
    return result;
}

static int image_tokens(const Transformer *model, const float *image,
                        size_t height, size_t width, float *tokens,
                        size_t *token_count) {
    if (!model || !image || !tokens || !token_count ||
        model->config.patch_size == 0 || model->config.input_channels == 0 ||
        model->patch_projection == NULL || height == 0 || width == 0 ||
        height % model->config.patch_size != 0 ||
        width % model->config.patch_size != 0) {
        return -1;
    }
    const size_t patch_size = model->config.patch_size;
    const size_t channels = model->config.input_channels;
    const size_t patches_y = height / patch_size;
    const size_t patches_x = width / patch_size;
    const size_t count = patches_y * patches_x;
    const size_t patch_elements = patch_size * patch_size * channels;
    float *patch = malloc(patch_elements * sizeof(float));
    if (!patch) {
        return -1;
    }
    size_t token = 0;
    for (size_t py = 0; py < patches_y; ++py) {
        for (size_t px = 0; px < patches_x; ++px) {
            size_t patch_index = 0;
            for (size_t y = 0; y < patch_size; ++y) {
                for (size_t x = 0; x < patch_size; ++x) {
                    const size_t pixel = ((py * patch_size + y) * width +
                                          px * patch_size + x) * channels;
                    for (size_t channel = 0; channel < channels; ++channel) {
                        patch[patch_index++] = image[pixel + channel];
                    }
                }
            }
            for (size_t dimension = 0; dimension < model->config.d_model; ++dimension) {
                float value = 0.0f;
                for (size_t i = 0; i < patch_elements; ++i) {
                    value += patch[i] *
                             model->patch_projection[dimension * patch_elements + i];
                }
                tokens[token * model->config.d_model + dimension] = value;
            }
            ++token;
        }
    }
    add_position_encoding(tokens, count, model->config.d_model);
    free(patch);
    *token_count = count;
    return 0;
}

float transformer_train_image(Transformer *model, const float *image,
                              size_t height, size_t width, size_t label,
                              float learning_rate) {
    if (!model || !model->classifier || model->config.n_classes == 0 ||
        label >= model->config.n_classes || learning_rate <= 0.0f ||
        model->config.patch_size == 0) {
        return -1.0f;
    }
    const size_t max_tokens = (height / model->config.patch_size) *
                              (width / model->config.patch_size);
    const size_t dimensions = model->config.d_model;
    float *tokens = calloc(max_tokens * dimensions, sizeof(float));
    float *encoded = calloc(max_tokens * dimensions, sizeof(float));
    float *pooled = calloc(dimensions, sizeof(float));
    float *probabilities = calloc(model->config.n_classes, sizeof(float));
    if (!tokens || !encoded || !pooled || !probabilities) {
        free(tokens);
        free(encoded);
        free(pooled);
        free(probabilities);
        return -1.0f;
    }
    size_t token_count = 0;
    if (image_tokens(model, image, height, width, tokens, &token_count) != 0 ||
        transformer_forward(model, tokens, token_count, encoded) != 0) {
        free(tokens);
        free(encoded);
        free(pooled);
        free(probabilities);
        return -1.0f;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            pooled[dimension] += encoded[token * dimensions + dimension] /
                                (float)token_count;
        }
    }
    float maximum = -INFINITY;
    for (size_t class_index = 0; class_index < model->config.n_classes; ++class_index) {
        float logit = 0.0f;
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            logit += model->classifier[class_index * dimensions + dimension] *
                     pooled[dimension];
        }
        probabilities[class_index] = logit;
        if (logit > maximum) {
            maximum = logit;
        }
    }
    float denominator = 0.0f;
    for (size_t class_index = 0; class_index < model->config.n_classes; ++class_index) {
        probabilities[class_index] = expf(probabilities[class_index] - maximum);
        denominator += probabilities[class_index];
    }
    for (size_t class_index = 0; class_index < model->config.n_classes; ++class_index) {
        probabilities[class_index] /= denominator;
    }
    const float loss = -logf(fmaxf(probabilities[label], 1e-12f));
    for (size_t class_index = 0; class_index < model->config.n_classes; ++class_index) {
        const float gradient = probabilities[class_index] -
                               (class_index == label ? 1.0f : 0.0f);
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            model->classifier[class_index * dimensions + dimension] -=
                learning_rate * gradient * pooled[dimension];
        }
    }
    free(tokens);
    free(encoded);
    free(pooled);
    free(probabilities);
    return loss;
}

float transformer_train_dataset(Transformer *model, const float *images,
                                const size_t *labels, size_t sample_count,
                                size_t height, size_t width, size_t epochs,
                                float learning_rate) {
    if (!model || !images || !labels || sample_count == 0 || epochs == 0 ||
        height == 0 || width == 0 || model->config.input_channels == 0) {
        return -1.0f;
    }
    const size_t image_elements = height * width * model->config.input_channels;
    float epoch_loss = 0.0f;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        epoch_loss = 0.0f;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const float loss = transformer_train_image(
                model, images + sample * image_elements, height, width,
                labels[sample], learning_rate);
            if (loss < 0.0f) {
                return -1.0f;
            }
            epoch_loss += loss;
        }
        epoch_loss /= (float)sample_count;
    }
    return epoch_loss;
}

int transformer_predict_image(const Transformer *model, const float *image,
                              size_t height, size_t width, size_t *class_index) {
    if (!model || !model->classifier || !class_index ||
        model->config.n_classes == 0 || model->config.patch_size == 0) {
        return -1;
    }
    const size_t max_tokens = (height / model->config.patch_size) *
                              (width / model->config.patch_size);
    const size_t dimensions = model->config.d_model;
    float *tokens = calloc(max_tokens * dimensions, sizeof(float));
    float *encoded = calloc(max_tokens * dimensions, sizeof(float));
    if (!tokens || !encoded) {
        free(tokens);
        free(encoded);
        return -1;
    }
    size_t token_count = 0;
    if (image_tokens(model, image, height, width, tokens, &token_count) != 0 ||
        transformer_forward(model, tokens, token_count, encoded) != 0) {
        free(tokens);
        free(encoded);
        return -1;
    }
    size_t best_class = 0;
    float best_logit = -INFINITY;
    for (size_t class_candidate = 0; class_candidate < model->config.n_classes;
         ++class_candidate) {
        float logit = 0.0f;
        for (size_t token = 0; token < token_count; ++token) {
            for (size_t dimension = 0; dimension < dimensions; ++dimension) {
                logit += model->classifier[class_candidate * dimensions + dimension] *
                         encoded[token * dimensions + dimension] /
                         (float)token_count;
            }
        }
        if (logit > best_logit) {
            best_logit = logit;
            best_class = class_candidate;
        }
    }
    *class_index = best_class;
    free(tokens);
    free(encoded);
    return 0;
}

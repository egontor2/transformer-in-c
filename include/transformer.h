#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stddef.h>

typedef struct {
    size_t d_model;
    size_t n_heads;
    size_t d_ff;
    size_t n_layers;
    size_t patch_size;
    size_t input_channels;
    size_t n_classes;
} TransformerConfig;

typedef struct Transformer Transformer;

/*
 * Creates an encoder-only Transformer.
 *
 * Inputs and outputs use row-major layout:
 *   [sequence_length][config.d_model]
 */
Transformer *transformer_create(TransformerConfig config, unsigned int seed);
void transformer_free(Transformer *model);

/*
 * Runs the encoder over `input` and writes `output`.
 * `length` must be greater than zero.
 * Returns 0 on success and -1 on invalid arguments or allocation failure.
 */
int transformer_forward(
    const Transformer *model,
    const float *input,
    size_t length,
    float *output
);

/*
 * Converts an image to non-overlapping patch tokens and runs the encoder.
 * The image uses row-major layout [height][width][channels], with values
 * normally normalized to [0, 1]. Width and height must be divisible by
 * config.patch_size. The output has
 * (height / patch_size) * (width / patch_size) * d_model elements.
 */
int transformer_forward_image(
    const Transformer *model,
    const float *image,
    size_t height,
    size_t width,
    float *output
);

/*
 * Trains the classification head with one labeled image using SGD.
 * The encoder remains frozen in this first training API. `label` must be
 * smaller than config.n_classes. Returns the cross-entropy loss, or -1.
 */
float transformer_train_image(
    Transformer *model,
    const float *image,
    size_t height,
    size_t width,
    size_t label,
    float learning_rate
);

/*
 * Trains the classification head over an in-memory dataset for `epochs`.
 * `images` is [sample_count][height][width][input_channels] and labels has
 * one class index per sample. Returns the mean final-epoch loss, or -1.
 */
float transformer_train_dataset(
    Transformer *model,
    const float *images,
    const size_t *labels,
    size_t sample_count,
    size_t height,
    size_t width,
    size_t epochs,
    float learning_rate
);

/* Predicts the class with the largest classification-head logit. */
int transformer_predict_image(
    const Transformer *model,
    const float *image,
    size_t height,
    size_t width,
    size_t *class_index
);

#endif

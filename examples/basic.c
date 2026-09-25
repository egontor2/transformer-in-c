#include "transformer.h"

#include <stdio.h>

int main(void) {
    const TransformerConfig config = {
        .d_model = 8,
        .n_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .patch_size = 2,
        .input_channels = 1,
        .n_classes = 2,
    };
    Transformer *model = transformer_create(config, 42);
    if (model == NULL) {
        return 1;
    }

    const float input[16] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    float output[16];
    const int result = transformer_forward(model, input, 2, output);
    if (result == 0) {
        printf("output[0] = %.6f\n", output[0]);
    }
    const float image[16] = {
        0.0f, 0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 0.9f,
        0.7f, 0.6f, 0.5f, 0.4f,
    };
    float image_output[32];
    const int image_result = transformer_forward_image(model, image, 4, 4,
                                                       image_output);
    if (image_result == 0) {
        printf("image patch token[0] = %.6f\n", image_output[0]);
    }
    for (size_t step = 0; step < 10; ++step) {
        const float loss = transformer_train_image(model, image, 4, 4, 1, 0.05f);
        if (loss < 0.0f) {
            transformer_free(model);
            return 1;
        }
        printf("train step %zu loss = %.6f\n", step, loss);
    }
    size_t prediction = 0;
    const int prediction_result = transformer_predict_image(
        model, image, 4, 4, &prediction);
    if (prediction_result == 0) {
        printf("predicted class = %zu\n", prediction);
    }
    transformer_free(model);
    return result != 0 || image_result != 0 || prediction_result != 0;
}

#include "transformer.h"
#include "dataset.h"
#include "autodiff.h"
#include "ops.h"
#include "arena.h"
#include "vit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

int main(void) {
    ViTPatchProjection patch_projection = {0};
    Tensor patch_tokens = {0};
    int vit_failures = vit_patch_projection_init(
        &patch_projection, 1, 2, 4, 2, 2) != 0 ||
        tensor_init(&patch_tokens, 2, 2) != 0;
    const float patch_image[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    if (!vit_failures) {
        patch_projection.projection.value.data[0] = 1.0f;
        patch_projection.projection.value.data[1] = 0.0f;
        patch_projection.projection.value.data[2] = 0.0f;
        patch_projection.projection.value.data[3] = 1.0f;
        patch_projection.projection.value.data[4] = 1.0f;
        patch_projection.projection.value.data[5] = 0.0f;
        patch_projection.projection.value.data[6] = 0.0f;
        patch_projection.projection.value.data[7] = 1.0f;
        vit_failures = vit_patch_projection_forward(
            &patch_projection, patch_image, 1, &patch_tokens) != 0 ||
            fabsf(patch_tokens.data[0] - 6.0f) > 1e-6f ||
            fabsf(patch_tokens.data[1] - 8.0f) > 1e-6f ||
            fabsf(patch_tokens.data[2] - 10.0f) > 1e-6f ||
            fabsf(patch_tokens.data[3] - 12.0f) > 1e-6f;
        patch_tokens.grad[0] = 1.0f;
        patch_tokens.grad[1] = 1.0f;
        patch_tokens.grad[2] = 1.0f;
        patch_tokens.grad[3] = 1.0f;
        float patch_image_grad[8] = {0};
        vit_failures = vit_failures ||
            vit_patch_projection_backward(&patch_projection, patch_image, 1,
                                          &patch_tokens, patch_image_grad) != 0 ||
            fabsf(patch_image_grad[0] - 1.0f) > 1e-6f ||
            fabsf(patch_image_grad[7] - 1.0f) > 1e-6f ||
            fabsf(patch_projection.bias.value.grad[0] - 2.0f) > 1e-6f;
    }
    tensor_free(&patch_tokens);
    vit_patch_projection_free(&patch_projection);
    if (vit_failures) {
        fprintf(stderr, "FAIL: ViT patch projection\n");
        return 1;
    }

    ViTTokenEmbedding token_embedding = {0};
    Tensor token_patch_input = {0};
    Tensor token_output = {0};
    Tensor token_patch_grad = {0};
    int token_failures = vit_token_embedding_init(
        &token_embedding, 2, 2) != 0 ||
        tensor_init(&token_patch_input, 2, 2) != 0 ||
        tensor_init(&token_output, 3, 2) != 0 ||
        tensor_init(&token_patch_grad, 2, 2) != 0;
    if (!token_failures) {
        token_patch_input.data[0] = 6.0f;
        token_patch_input.data[1] = 8.0f;
        token_patch_input.data[2] = 10.0f;
        token_patch_input.data[3] = 12.0f;
        token_embedding.cls_token.value.data[0] = 10.0f;
        token_embedding.cls_token.value.data[1] = 20.0f;
        token_embedding.positional_embeddings.value.data[0] = 1.0f;
        token_embedding.positional_embeddings.value.data[1] = 2.0f;
        token_embedding.positional_embeddings.value.data[2] = 3.0f;
        token_embedding.positional_embeddings.value.data[3] = 4.0f;
        token_failures = vit_token_embedding_forward(
            &token_embedding, &token_patch_input, 1, &token_output) != 0 ||
            fabsf(token_output.data[0] - 11.0f) > 1e-6f ||
            fabsf(token_output.data[1] - 22.0f) > 1e-6f ||
            fabsf(token_output.data[2] - 9.0f) > 1e-6f ||
            fabsf(token_output.data[3] - 12.0f) > 1e-6f;
        for (size_t i = 0; i < 6; ++i) {
            token_output.grad[i] = 1.0f;
        }
        token_failures = token_failures ||
            vit_token_embedding_backward(&token_embedding, &token_output, 1,
                                         &token_patch_grad) != 0 ||
            fabsf(token_embedding.cls_token.value.grad[0] - 1.0f) > 1e-6f ||
            fabsf(token_embedding.positional_embeddings.value.grad[0] - 1.0f) > 1e-6f ||
            fabsf(token_embedding.positional_embeddings.value.grad[2] - 1.0f) > 1e-6f ||
            fabsf(token_patch_grad.grad[0] - 1.0f) > 1e-6f;
    }
    tensor_free(&token_patch_input);
    tensor_free(&token_output);
    tensor_free(&token_patch_grad);
    vit_token_embedding_free(&token_embedding);
    if (token_failures) {
        fprintf(stderr, "FAIL: ViT CLS and positional embeddings\n");
        return 1;
    }

    Tensor query = {0};
    Tensor key = {0};
    Tensor value = {0};
    Tensor probabilities = {0};
    Tensor attended = {0};
    Tensor query_grad = {0};
    Tensor key_grad = {0};
    Tensor value_grad = {0};
    int attention_failures = tensor_init(&query, 2, 2) != 0 ||
        tensor_init(&key, 2, 2) != 0 || tensor_init(&value, 2, 1) != 0 ||
        tensor_init(&probabilities, 2, 2) != 0 || tensor_init(&attended, 2, 1) != 0 ||
        tensor_init(&query_grad, 2, 2) != 0 || tensor_init(&key_grad, 2, 2) != 0 ||
        tensor_init(&value_grad, 2, 1) != 0;
    if (!attention_failures) {
        query.data[0] = 1.0f;
        query.data[3] = 1.0f;
        key.data[0] = 1.0f;
        key.data[3] = 1.0f;
        value.data[0] = 2.0f;
        value.data[1] = 4.0f;
        attention_failures = ops_causal_attention(
            &query, &key, &value, 1.0f, &probabilities, &attended) != 0 ||
            fabsf(probabilities.data[1]) > 1e-6f ||
            probabilities.data[2] < 0.2f || probabilities.data[2] > 0.4f ||
            probabilities.data[3] < 0.6f || probabilities.data[3] > 0.8f;
        attended.grad[0] = 1.0f;
        attention_failures = attention_failures ||
            ops_causal_attention_backward(&query, &key, &value, 1.0f,
                                          &probabilities, &attended,
                                          &query_grad, &key_grad, &value_grad) != 0 ||
            !isfinite(query_grad.grad[0]);
    }
    tensor_free(&query);
    tensor_free(&key);
    tensor_free(&value);
    tensor_free(&probabilities);
    tensor_free(&attended);
    tensor_free(&query_grad);
    tensor_free(&key_grad);
    tensor_free(&value_grad);
    if (attention_failures) {
        fprintf(stderr, "FAIL: causal attention\n");
        return 1;
    }

    Arena arena = {0};
    Tensor arena_tensor = {0};
    const size_t arena_shape[3] = {2, 2, 2};
    int phase_two_failures = arena_init(&arena, 1024) != 0 ||
                             tensor_init_arena(&arena_tensor, &arena, 3,
                                               arena_shape, 1) != 0;
    if (!phase_two_failures) {
        phase_two_failures = tensor_numel(&arena_tensor) != 8;
        tensor_zero_grad(&arena_tensor);
    }
    tensor_free(&arena_tensor);
    arena_free(&arena);
    Tensor logits = {0};
    Tensor logits_grad = {0};
    if (!phase_two_failures) {
        phase_two_failures = tensor_init(&logits, 1, 2) != 0 ||
                             tensor_init(&logits_grad, 1, 2) != 0;
        logits.data[0] = 2.0f;
        logits.data[1] = 0.0f;
        const size_t target = 0;
        float loss = 0.0f;
        phase_two_failures = phase_two_failures ||
            ops_softmax_cross_entropy(&logits, &target, &loss, &logits_grad) != 0 ||
            !(loss > 0.0f && loss < 0.2f) ||
            !(logits_grad.grad[0] < 0.0f && logits_grad.grad[1] > 0.0f);
    }
    tensor_free(&logits);
    tensor_free(&logits_grad);
    Parameter adam = {0};
    if (!phase_two_failures) {
        phase_two_failures = parameter_init(&adam, 1, 1) != 0;
        adam.value.data[0] = 1.0f;
        adam.value.grad[0] = 1.0f;
        phase_two_failures = phase_two_failures ||
            ops_adamw_step(&adam, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f) != 0 ||
            !(adam.value.data[0] < 1.0f) || adam.step != 1;
    }
    parameter_free(&adam);
    if (phase_two_failures) {
        fprintf(stderr, "FAIL: phase 2 primitives\n");
        return 1;
    }

    Tensor gemm_left = {0};
    Tensor gemm_right = {0};
    Tensor gemm_output = {0};
    int ops_failures = tensor_init(&gemm_left, 2, 2) != 0 ||
                       tensor_init(&gemm_right, 2, 2) != 0 ||
                       tensor_init(&gemm_output, 2, 2) != 0;
    if (!ops_failures) {
        gemm_left.data[0] = 1.0f;
        gemm_left.data[3] = 1.0f;
        gemm_right.data[0] = 2.0f;
        gemm_right.data[3] = 3.0f;
        ops_failures = ops_gemm(&gemm_left, &gemm_right, &gemm_output) != 0 ||
                        fabsf(gemm_output.data[0] - 2.0f) > 1e-6f ||
                        fabsf(gemm_output.data[3] - 3.0f) > 1e-6f;
        gemm_output.grad[0] = 1.0f;
        gemm_output.grad[3] = 1.0f;
        ops_failures = ops_failures ||
                        ops_gemm_backward(&gemm_left, &gemm_right,
                                          &gemm_output) != 0 ||
                        fabsf(gemm_left.grad[0] - 2.0f) > 1e-6f ||
                        fabsf(gemm_right.grad[3] - 1.0f) > 1e-6f;
    }
    tensor_free(&gemm_left);
    tensor_free(&gemm_right);
    tensor_free(&gemm_output);
    if (ops_failures) {
        fprintf(stderr, "FAIL: phase 1 GEMM operations\n");
        return 1;
    }

    Tensor left = {0};
    Tensor right = {0};
    Tensor product = {0};
    int autodiff_failures = tensor_init(&left, 1, 2) != 0 ||
                            tensor_init(&right, 2, 1) != 0 ||
                            tensor_init(&product, 1, 1) != 0;
    if (!autodiff_failures) {
        left.data[0] = 2.0f;
        left.data[1] = 3.0f;
        right.data[0] = 4.0f;
        right.data[1] = 5.0f;
        tensor_matmul(&left, &right, &product);
        product.grad[0] = 1.0f;
        tensor_matmul_backward(&left, &right, &product);
        autodiff_failures = fabsf(product.data[0] - 23.0f) > 1e-6f ||
                            fabsf(left.grad[0] - 4.0f) > 1e-6f ||
                            fabsf(right.grad[1] - 3.0f) > 1e-6f;
    }
    tensor_free(&left);
    tensor_free(&right);
    tensor_free(&product);
    if (autodiff_failures) {
        fprintf(stderr, "FAIL: autodiff matmul\n");
        return 1;
    }

    const char *image_path = "build/test_image.pgm";
    const char *manifest_path = "build/test_manifest.csv";
    FILE *image_file = fopen(image_path, "w");
    FILE *manifest_file = fopen(manifest_path, "w");
    if (!image_file || !manifest_file) {
        if (image_file) fclose(image_file);
        if (manifest_file) fclose(manifest_file);
        return 1;
    }
    fprintf(image_file, "P2\n2 2\n255\n0 64 128 255\n");
    fprintf(manifest_file, "%s,3\n%s,4\n", image_path, image_path);
    fclose(image_file);
    fclose(manifest_file);
    VisionDataset dataset;
    int dataset_result = vision_dataset_load_pgm_csv(manifest_path, &dataset);
    int dataset_failures = 0;
    dataset_failures += expect(dataset_result == 0, "PGM manifest loading");
    if (dataset_result == 0) {
        dataset_failures += expect(dataset.sample_count == 2 &&
                                    dataset.height == 2 &&
                                    dataset.width == 2 &&
                                    dataset.channels == 1,
                                    "dataset dimensions");
        dataset_failures += expect(fabsf(dataset.images[3] - 1.0f) < 1e-6f,
                                    "dataset pixel normalization");
        dataset_failures += expect(dataset.labels[0] == 3 && dataset.labels[1] == 4,
                                    "dataset labels");
        vision_dataset_free(&dataset);
    }
    remove(image_path);
    remove(manifest_path);
    if (dataset_failures != 0) {
        return 1;
    }

    const TransformerConfig config = {
        .d_model = 8,
        .n_heads = 2,
        .d_ff = 16,
        .n_layers = 2,
        .patch_size = 2,
        .input_channels = 1,
        .n_classes = 2,
    };
    const float image[16] = {
        0.0f, 0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 0.9f,
        0.7f, 0.6f, 0.5f, 0.4f,
    };
    Transformer *first = transformer_create(config, 7);
    Transformer *second = transformer_create(config, 7);
    if (expect(first != NULL && second != NULL, "model creation") != 0) {
        transformer_free(first);
        transformer_free(second);
        return 1;
    }

    float first_output[32];
    float second_output[32];
    int failures = 0;
    failures += expect(transformer_forward_image(first, image, 4, 4,
                                                 first_output) == 0,
                       "valid image forward");
    failures += expect(transformer_forward_image(second, image, 4, 4,
                                                 second_output) == 0,
                       "second valid image forward");
    for (size_t i = 0; i < 32; ++i) {
        failures += expect(fabsf(first_output[i] - second_output[i]) < 1e-6f,
                           "seeded models are deterministic");
    }

    float invalid_output[32];
    failures += expect(transformer_forward_image(first, image, 3, 4,
                                                 invalid_output) != 0,
                       "rejects non-divisible image height");
    failures += expect(transformer_forward_image(first, image, 4, 3,
                                                 invalid_output) != 0,
                       "rejects non-divisible image width");

    const float initial_loss = transformer_train_image(
        first, image, 4, 4, 1, 0.1f);
    float final_loss = initial_loss;
    for (size_t step = 0; step < 100; ++step) {
        final_loss = transformer_train_image(first, image, 4, 4, 1, 0.1f);
    }
    failures += expect(initial_loss > 0.0f && final_loss < initial_loss,
                       "classifier training reduces loss");

    size_t prediction = 0;
    failures += expect(transformer_predict_image(first, image, 4, 4,
                                                 &prediction) == 0,
                       "image prediction");
    failures += expect(prediction == 1, "trained classifier predicts label");

    Transformer *dataset_model = transformer_create(config, 11);
    const float dataset_images[32] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f, 0.9f, 0.7f, 0.6f, 0.5f, 0.4f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 0.9f, 0.8f, 0.6f, 0.5f, 0.4f, 0.3f,
    };
    const size_t labels[2] = {1, 1};
    failures += expect(dataset_model != NULL, "dataset model creation");
    const float dataset_loss = transformer_train_dataset(
        dataset_model, dataset_images, labels, 2, 4, 4, 5, 0.1f);
    failures += expect(dataset_loss > 0.0f, "dataset training");
    transformer_free(dataset_model);

    transformer_free(first);
    transformer_free(second);
    return failures == 0 ? 0 : 1;
}

#include "transformer.h"
#include "augmentation.h"
#include "dataset.h"
#include "autodiff.h"
#include "ops.h"
#include "arena.h"
#include "vit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

int main(void) {
    ViTClassificationHead classification_head = {0};
    Tensor classification_tokens = {0};
    Tensor classification_logits = {0};
    int classification_failures = vit_classification_head_init(
        &classification_head, 2, 2) != 0 ||
        tensor_init(&classification_tokens, 3, 2) != 0 ||
        tensor_init(&classification_logits, 1, 2) != 0;
    if (!classification_failures) {
        classification_tokens.data[0] = 2.0f;
        classification_tokens.data[1] = 3.0f;
        classification_head.projection.value.data[0] = 1.0f;
        classification_head.projection.value.data[1] = 2.0f;
        classification_head.projection.value.data[2] = 3.0f;
        classification_head.projection.value.data[3] = 4.0f;
        classification_failures =
            vit_classification_head_forward(&classification_head,
                                            &classification_tokens, 1, 3,
                                            &classification_logits) != 0 ||
            fabsf(classification_logits.data[0] - 11.0f) > 1e-6f ||
            fabsf(classification_logits.data[1] - 16.0f) > 1e-6f;
        classification_logits.grad[0] = 1.0f;
        classification_logits.grad[1] = -1.0f;
        classification_failures = classification_failures ||
            vit_classification_head_backward(&classification_head,
                                             &classification_tokens, 1, 3,
                                             &classification_logits,
                                             &classification_tokens) != 0 ||
            fabsf(classification_head.bias.value.grad[0] - 1.0f) > 1e-6f ||
            fabsf(classification_tokens.grad[0] + 1.0f) > 1e-6f;
    }
    tensor_free(&classification_tokens);
    tensor_free(&classification_logits);
    vit_classification_head_free(&classification_head);
    if (classification_failures) {
        fprintf(stderr, "FAIL: ViT classification head\n");
        return 1;
    }

    ViTEncoderBlock encoder_block = {0};
    Tensor block_input = {0};
    Tensor block_output = {0};
    int block_failures = vit_encoder_block_init(
        &encoder_block, 4, 2, 3) != 0 ||
        tensor_init(&block_input, 3, 4) != 0 ||
        tensor_init(&block_output, 3, 4) != 0;
    if (!block_failures) {
        for (size_t i = 0; i < tensor_numel(&block_input); ++i) {
            block_input.data[i] = (float)(i + 1) * 0.1f;
        }
        block_failures = vit_encoder_block_forward(
            &encoder_block, &block_input, 1, &block_output) != 0;
        for (size_t i = 0; i < tensor_numel(&block_output); ++i) {
            block_failures = block_failures || !isfinite(block_output.data[i]);
        }
    }
    tensor_free(&block_input);
    tensor_free(&block_output);
    vit_encoder_block_free(&encoder_block);
    if (block_failures) {
        fprintf(stderr, "FAIL: ViT encoder block forward\n");
        return 1;
    }

    ViTEncoderBlock cached_block = {0};
    ViTEncoderBlockCache block_cache = {0};
    Tensor cached_input = {0};
    Tensor cached_input_grad = {0};
    int cached_failures = vit_encoder_block_init(
        &cached_block, 4, 2, 3) != 0 ||
        tensor_init(&cached_input, 3, 4) != 0 ||
        tensor_init(&cached_input_grad, 3, 4) != 0 ||
        vit_encoder_block_cache_init(&block_cache, &cached_block, 1) != 0;
    if (!cached_failures) {
        for (size_t i = 0; i < tensor_numel(&cached_input); ++i) {
            cached_input.data[i] = (float)(i + 1) * 0.07f;
        }
        cached_failures =
            vit_encoder_block_forward_cached(&cached_block, &cached_input,
                                             &block_cache) != 0;
        for (size_t i = 0; i < tensor_numel(&block_cache.output); ++i) {
            block_cache.output.grad[i] = 1.0f;
        }
        cached_failures = cached_failures ||
            vit_encoder_block_backward(&cached_block, &cached_input,
                                       &block_cache, &cached_input_grad) != 0;
        for (size_t i = 0; i < tensor_numel(&cached_input_grad); ++i) {
            cached_failures = cached_failures ||
                !isfinite(cached_input_grad.grad[i]);
        }
        cached_failures = cached_failures ||
            !isfinite(cached_block.query_key_value.value.grad[0]) ||
            !isfinite(cached_block.mlp_output.value.grad[0]);
        const float step = 1e-3f;
        const float original = cached_input.data[0];
        cached_input.data[0] = original + step;
        vit_encoder_block_forward_cached(&cached_block, &cached_input,
                                         &block_cache);
        float positive = 0.0f;
        for (size_t i = 0; i < tensor_numel(&block_cache.output); ++i) {
            positive += block_cache.output.data[i];
        }
        cached_input.data[0] = original - step;
        vit_encoder_block_forward_cached(&cached_block, &cached_input,
                                         &block_cache);
        float negative = 0.0f;
        for (size_t i = 0; i < tensor_numel(&block_cache.output); ++i) {
            negative += block_cache.output.data[i];
        }
        cached_input.data[0] = original;
        cached_failures = cached_failures ||
            fabsf((positive - negative) / (2.0f * step) -
                  cached_input_grad.grad[0]) > 2e-2f;
    }
    tensor_free(&cached_input);
    tensor_free(&cached_input_grad);
    vit_encoder_block_cache_free(&block_cache);
    vit_encoder_block_free(&cached_block);
    if (cached_failures) {
        fprintf(stderr, "FAIL: ViT encoder block cached backward\n");
        return 1;
    }

    ViTEncoderBlock qkv_block = {0};
    ViTEncoderBlockCache qkv_cache = {0};
    Tensor qkv_input = {0};
    Tensor qkv_input_grad = {0};
    int qkv_failures = vit_encoder_block_init(&qkv_block, 8, 2, 5) != 0 ||
        tensor_init(&qkv_input, 5, 8) != 0 ||
        tensor_init(&qkv_input_grad, 5, 8) != 0 ||
        vit_encoder_block_cache_init(&qkv_cache, &qkv_block, 1) != 0;
    if (!qkv_failures) {
        for (size_t i = 0; i < tensor_numel(&qkv_block.query_key_value.value); ++i) {
            qkv_block.query_key_value.value.data[i] = sinf((float)i * 1.3f) * 0.5f;
        }
        for (size_t i = 0; i < tensor_numel(&qkv_block.attention_output.value); ++i) {
            qkv_block.attention_output.value.data[i] = cosf((float)i * 0.7f) * 0.5f;
        }
        for (size_t i = 0; i < tensor_numel(&qkv_input); ++i) {
            qkv_input.data[i] = sinf((float)i * 0.37f);
        }
        qkv_failures = vit_encoder_block_forward_cached(&qkv_block, &qkv_input,
                                                        &qkv_cache) != 0;
        for (size_t i = 0; i < tensor_numel(&qkv_cache.output); ++i) {
            qkv_cache.output.grad[i] = 1.0f;
        }
        qkv_failures = qkv_failures ||
            vit_encoder_block_backward(&qkv_block, &qkv_input, &qkv_cache,
                                       &qkv_input_grad) != 0;
        const size_t indices[] = {0, 17, 40, 90};
        for (size_t k = 0; !qkv_failures && k < 4; ++k) {
            float *weight = &qkv_block.query_key_value.value.data[indices[k]];
            const float original = *weight;
            const float step = 1e-3f;
            float sums[2] = {0.0f, 0.0f};
            for (int side = 0; side < 2; ++side) {
                *weight = original + (side == 0 ? step : -step);
                vit_encoder_block_forward_cached(&qkv_block, &qkv_input,
                                                 &qkv_cache);
                for (size_t i = 0; i < tensor_numel(&qkv_cache.output); ++i) {
                    sums[side] += qkv_cache.output.data[i];
                }
            }
            *weight = original;
            const float numeric = (sums[0] - sums[1]) / (2.0f * step);
            qkv_failures = fabsf(numeric -
                qkv_block.query_key_value.value.grad[indices[k]]) > 2e-2f;
        }
    }
    tensor_free(&qkv_input);
    tensor_free(&qkv_input_grad);
    vit_encoder_block_cache_free(&qkv_cache);
    vit_encoder_block_free(&qkv_block);
    if (qkv_failures) {
        fprintf(stderr, "FAIL: ViT encoder block QKV gradient\n");
        return 1;
    }

    ViTConfig model_config = {1, 4, 4, 2, 4, 2, 1, 2};
    ViTModel model = {0};
    ViTModelCache model_cache = {0};
    float model_image[16];
    float model_image_grad[16] = {0};
    int model_failures = vit_model_init(&model, &model_config) != 0;
    if (!model_failures) {
        for (size_t i = 0; i < 16; ++i) {
            model_image[i] = (float)(i + 1) * 0.05f;
        }
        model_failures =
            vit_model_cache_init(&model_cache, &model, 1) != 0 ||
            vit_model_forward(&model, model_image, 1, &model_cache) != 0;
        if (!model_failures) {
            model_cache.logits.grad[0] = 1.0f;
            model_cache.logits.grad[1] = -1.0f;
            model_failures = vit_model_backward(
                &model, model_image, &model_cache, model_image_grad) != 0;
            for (size_t i = 0; i < 16; ++i) {
                model_failures = model_failures || !isfinite(model_image_grad[i]);
            }
        }
    }
    vit_model_cache_free(&model_cache);
    vit_model_free(&model);
    if (model_failures) {
        fprintf(stderr, "FAIL: integrated ViT model\n");
        return 1;
    }

    ViTModel seeded_first = {0};
    ViTModel seeded_second = {0};
    ViTModel seeded_other = {0};
    int seed_failures =
        vit_model_init_seeded(&seeded_first, &model_config, 17) != 0 ||
        vit_model_init_seeded(&seeded_second, &model_config, 17) != 0 ||
        vit_model_init_seeded(&seeded_other, &model_config, 18) != 0;
    if (!seed_failures) {
        const Tensor *first = &seeded_first.patch_projection.projection.value;
        const Tensor *second = &seeded_second.patch_projection.projection.value;
        const Tensor *other = &seeded_other.patch_projection.projection.value;
        int differs = 0;
        for (size_t i = 0; i < tensor_numel(first); ++i) {
            seed_failures = seed_failures ||
                first->data[i] != second->data[i];
            differs = differs || first->data[i] != other->data[i];
        }
        seed_failures = seed_failures || !differs;
    }
    vit_model_free(&seeded_first);
    vit_model_free(&seeded_second);
    vit_model_free(&seeded_other);
    if (seed_failures) {
        fprintf(stderr, "FAIL: seeded ViT initialization\n");
        return 1;
    }

    ViTConfig training_config = {1, 2, 2, 1, 2, 1, 1, 2};
    ViTModel training_model = {0};
    ViTModelCache training_cache = {0};
    const float training_images[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const size_t training_targets[1] = {0};
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    int training_failures = vit_model_init(&training_model, &training_config) != 0 ||
        vit_model_cache_init(&training_cache, &training_model, 1) != 0;
    if (!training_failures) {
        for (size_t step = 0; step < 50; ++step) {
            float loss = 0.0f;
            training_failures = vit_model_train_batch(
                &training_model, training_images, training_targets, 1,
                &training_cache, 0.1f, 0.001f, &loss) != 0;
            if (step == 0) {
                first_loss = loss;
            }
            last_loss = loss;
            if (training_failures) {
                break;
            }
        }
        training_failures = training_failures || !isfinite(first_loss) ||
            !isfinite(last_loss) || last_loss >= first_loss;
    }
    if (training_failures) {
        vit_model_cache_free(&training_cache);
        vit_model_free(&training_model);
        fprintf(stderr, "FAIL: ViT end-to-end training\n");
        return 1;
    }

    const char *checkpoint_path = "build/test_vit_checkpoint.bin";
    ViTModel restored_model = {0};
    ViTModelCache restored_cache = {0};
    int checkpoint_failures = vit_model_save(&training_model, checkpoint_path) != 0;
    checkpoint_failures = checkpoint_failures ||
        vit_model_init(&restored_model, &training_config) != 0;
    checkpoint_failures = checkpoint_failures ||
        vit_model_load(&restored_model, checkpoint_path) != 0;
    checkpoint_failures = checkpoint_failures ||
        vit_model_cache_init(&restored_cache, &restored_model, 1) != 0 ||
        vit_model_forward(&training_model, training_images, 1,
                          &training_cache) != 0 ||
        vit_model_forward(&restored_model, training_images, 1,
                          &restored_cache) != 0;
    if (!checkpoint_failures) {
        for (size_t i = 0; i < tensor_numel(&training_cache.logits); ++i) {
            checkpoint_failures = checkpoint_failures ||
                fabsf(training_cache.logits.data[i] -
                      restored_cache.logits.data[i]) > 1e-7f;
        }
        checkpoint_failures = checkpoint_failures ||
            training_model.classification_head.projection.step !=
                restored_model.classification_head.projection.step;
    }
    ViTEvaluationMetrics evaluation = {0};
    int evaluation_failures = vit_model_evaluate(
        &training_model, training_images, training_targets, 1, &evaluation) != 0 ||
        !isfinite(evaluation.accuracy) ||
        !isfinite(evaluation.average_loss) ||
        evaluation.accuracy < 0.0f || evaluation.accuracy > 1.0f ||
        evaluation.samples_per_second < 0.0;
    vit_model_cache_free(&restored_cache);
    vit_model_free(&restored_model);
    vit_model_cache_free(&training_cache);
    vit_model_free(&training_model);
    remove(checkpoint_path);
    if (checkpoint_failures) {
        fprintf(stderr, "FAIL: ViT checkpoint round-trip\n");
        return 1;
    }
    if (evaluation_failures) {
        fprintf(stderr, "FAIL: ViT evaluation\n");
        return 1;
    }

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

    ViTPatchProjection gradient_projection = {0};
    Tensor gradient_output = {0};
    int gradient_failures = vit_patch_projection_init(
        &gradient_projection, 1, 2, 2, 2, 1) != 0 ||
        tensor_init(&gradient_output, 1, 1) != 0;
    const float gradient_input[4] = {1.0f, -2.0f, 0.5f, 3.0f};
    if (!gradient_failures) {
        gradient_projection.projection.value.data[0] = 0.2f;
        gradient_projection.projection.value.data[1] = -0.4f;
        gradient_projection.projection.value.data[2] = 0.7f;
        gradient_projection.projection.value.data[3] = 0.1f;
        gradient_output.grad[0] = 1.0f;
        float analytic_input_grad[4] = {0};
        gradient_failures =
            vit_patch_projection_forward(&gradient_projection, gradient_input,
                                          1, &gradient_output) != 0 ||
            vit_patch_projection_backward(&gradient_projection, gradient_input,
                                          1, &gradient_output,
                                          analytic_input_grad) != 0;
        const float finite_difference = 1e-3f;
        const float original_weight =
            gradient_projection.projection.value.data[0];
        gradient_projection.projection.value.data[0] =
            original_weight + finite_difference;
        vit_patch_projection_forward(&gradient_projection, gradient_input, 1,
                                     &gradient_output);
        const float positive = gradient_output.data[0];
        gradient_projection.projection.value.data[0] =
            original_weight - finite_difference;
        vit_patch_projection_forward(&gradient_projection, gradient_input, 1,
                                     &gradient_output);
        const float negative = gradient_output.data[0];
        gradient_projection.projection.value.data[0] = original_weight;
        const float numerical_weight =
            (positive - negative) / (2.0f * finite_difference);
        const float weight_error =
            fabsf(numerical_weight -
                  gradient_projection.projection.value.grad[0]);
        const float original_pixel = gradient_input[0];
        float positive_input[4];
        float negative_input[4];
        memcpy(positive_input, gradient_input, sizeof(gradient_input));
        memcpy(negative_input, gradient_input, sizeof(gradient_input));
        positive_input[0] = original_pixel + finite_difference;
        negative_input[0] = original_pixel - finite_difference;
        vit_patch_projection_forward(&gradient_projection, positive_input, 1,
                                     &gradient_output);
        const float positive_pixel = gradient_output.data[0];
        vit_patch_projection_forward(&gradient_projection, negative_input, 1,
                                     &gradient_output);
        const float negative_pixel = gradient_output.data[0];
        const float numerical_pixel =
            (positive_pixel - negative_pixel) / (2.0f * finite_difference);
        gradient_failures = gradient_failures ||
            weight_error > 1e-3f ||
            fabsf(numerical_pixel - analytic_input_grad[0]) > 1e-3f;
    }
    tensor_free(&gradient_output);
    vit_patch_projection_free(&gradient_projection);
    if (gradient_failures) {
        fprintf(stderr, "FAIL: ViT patch projection gradient check\n");
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

    Tensor multi_query = {0};
    Tensor multi_key = {0};
    Tensor multi_value = {0};
    Tensor multi_probabilities = {0};
    Tensor multi_output = {0};
    Tensor multi_query_grad = {0};
    Tensor multi_key_grad = {0};
    Tensor multi_value_grad = {0};
    int multi_head_failures = tensor_init(&multi_query, 2, 2) != 0 ||
        tensor_init(&multi_key, 2, 2) != 0 ||
        tensor_init(&multi_value, 2, 1) != 0 ||
        tensor_init(&multi_probabilities, 2, 2) != 0 ||
        tensor_init(&multi_output, 2, 1) != 0 ||
        tensor_init(&multi_query_grad, 2, 2) != 0 ||
        tensor_init(&multi_key_grad, 2, 2) != 0 ||
        tensor_init(&multi_value_grad, 2, 1) != 0;
    if (!multi_head_failures) {
        multi_query.data[0] = 1.0f;
        multi_query.data[3] = 1.0f;
        multi_key.data[0] = 1.0f;
        multi_key.data[3] = 1.0f;
        multi_value.data[0] = 2.0f;
        multi_value.data[1] = 4.0f;
        multi_head_failures =
            ops_multi_head_attention(&multi_query, &multi_key, &multi_value,
                                     1, 1, 2, 1.0f, &multi_probabilities,
                                     &multi_output) != 0 ||
            multi_probabilities.data[0] < 0.6f ||
            multi_probabilities.data[1] < 0.2f ||
            fabsf(multi_probabilities.data[0] +
                  multi_probabilities.data[1] - 1.0f) > 1e-6f;
        multi_output.grad[0] = 1.0f;
        multi_output.grad[1] = 1.0f;
        multi_head_failures = multi_head_failures ||
            ops_multi_head_attention_backward(
                &multi_query, &multi_key, &multi_value, 1, 1, 2, 1.0f,
                &multi_probabilities, &multi_output, &multi_query_grad,
                &multi_key_grad, &multi_value_grad) != 0 ||
            !isfinite(multi_query_grad.grad[0]) ||
            !isfinite(multi_key_grad.grad[0]) ||
            !isfinite(multi_value_grad.grad[0]);
    }
    tensor_free(&multi_query);
    tensor_free(&multi_key);
    tensor_free(&multi_value);
    tensor_free(&multi_probabilities);
    tensor_free(&multi_output);
    tensor_free(&multi_query_grad);
    tensor_free(&multi_key_grad);
    tensor_free(&multi_value_grad);
    if (multi_head_failures) {
        fprintf(stderr, "FAIL: bidirectional multi-head attention\n");
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

    Tensor smoothed_logits = {0};
    const size_t smoothed_targets[2] = {0, 2};
    const float label_smoothing = 0.1f;
    int smoothing_failures = tensor_init(&smoothed_logits, 2, 3) != 0;
    if (!smoothing_failures) {
        const float logits_values[6] = {2.0f, -1.0f, 0.5f, 0.0f, 1.0f, 3.0f};
        memcpy(smoothed_logits.data, logits_values, sizeof(logits_values));
        float smoothed_loss = 0.0f;
        smoothing_failures = ops_softmax_cross_entropy_smoothed(
            &smoothed_logits, smoothed_targets, NULL, label_smoothing,
            &smoothed_loss, &smoothed_logits) != 0;
        float expected_loss = 0.0f;
        for (size_t row = 0; row < 2; ++row) {
            const float *row_logits = logits_values + row * 3;
            float normalizer = 0.0f;
            for (size_t col = 0; col < 3; ++col) {
                normalizer += expf(row_logits[col]);
            }
            float gradient_sum = 0.0f;
            for (size_t col = 0; col < 3; ++col) {
                const float target_probability =
                    (col == smoothed_targets[row] ? 1.0f - label_smoothing : 0.0f) +
                    label_smoothing / 3.0f;
                expected_loss -= target_probability *
                    (row_logits[col] - logf(normalizer)) / 2.0f;
                gradient_sum += smoothed_logits.grad[row * 3 + col];
            }
            smoothing_failures = smoothing_failures || fabsf(gradient_sum) > 1e-6f;
        }
        smoothing_failures = smoothing_failures ||
            fabsf(smoothed_loss - expected_loss) > 1e-5f;
    }
    tensor_free(&smoothed_logits);
    failures += expect(!smoothing_failures, "label smoothing cross entropy");

    float augmentation_source[2 * 16];
    float augmentation_output[2 * 16];
    for (size_t i = 0; i < 2 * 16; ++i) {
        augmentation_source[i] = (float)(i % 16) / 15.0f;
    }
    uint64_t augmentation_random_state = 1;
    DataAugmentation identity_augmentation = {0};
    int augmentation_failures = data_augmentation_enabled(&identity_augmentation) ||
        data_augmentation_apply(&identity_augmentation,
                                &augmentation_random_state,
                                augmentation_source, augmentation_output,
                                2, 1, 4, 4) != 0 ||
        memcmp(augmentation_source, augmentation_output,
               sizeof(augmentation_source)) != 0;
    DataAugmentation full_augmentation = {
        .max_shift_pixels = 1.0f,
        .max_rotation_degrees = 15.0f,
        .max_scale_delta = 0.1f,
        .max_brightness_delta = 0.1f,
        .max_contrast_delta = 0.2f,
        .noise_standard_deviation = 0.05f,
        .erasing_probability = 1.0f,
        .erasing_max_side_fraction = 0.5f,
        .erasing_value = 0.5f,
        .minimum_value = 0.0f,
        .maximum_value = 1.0f,
    };
    augmentation_failures = augmentation_failures ||
        !data_augmentation_enabled(&full_augmentation) ||
        data_augmentation_apply(&full_augmentation, &augmentation_random_state,
                                augmentation_source, augmentation_output,
                                2, 1, 4, 4) != 0 ||
        data_augmentation_apply(&full_augmentation, &augmentation_random_state,
                                augmentation_source, augmentation_source,
                                2, 1, 4, 4) == 0;
    for (size_t i = 0; i < 2 * 16; ++i) {
        augmentation_failures = augmentation_failures ||
            !isfinite(augmentation_output[i]) ||
            augmentation_output[i] < 0.0f || augmentation_output[i] > 1.0f;
    }
    failures += expect(!augmentation_failures, "data augmentation");

    transformer_free(first);
    transformer_free(second);
    return failures == 0 ? 0 : 1;
}

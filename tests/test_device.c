#if defined(ENABLE_CUDA)
#include "cuda_backend.h"
typedef CudaBackend DeviceBackend;
#define DEVICE_NAME "CUDA"
#define device_backend_create cuda_backend_create
#define device_backend_available cuda_backend_available
#define device_backend_enable cuda_backend_enable_device_execution
#define device_backend_disable cuda_backend_disable_device_execution
#define device_backend_free cuda_backend_free
#else
#include "mps_backend.h"
typedef MPSBackend DeviceBackend;
#define DEVICE_NAME "Metal"
#define device_backend_create mps_backend_create
#define device_backend_available mps_backend_available
#define device_backend_enable mps_backend_enable_device_execution
#define device_backend_disable mps_backend_disable_device_execution
#define device_backend_free mps_backend_free
#endif
#include "ops.h"
#include "vit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BATCH 8
#define MAX_TRAINING_STEPS 12
#define IMAGE_SIZE (28 * 28)

typedef struct {
    const char *name;
    ViTConfig config;
    size_t batch;
    size_t training_steps;
} TestScenario;

static void fill_batch(float *images, size_t *labels, size_t batch) {
    for (size_t i = 0; i < batch * IMAGE_SIZE; ++i) {
        images[i] = sinf((float)i * 0.013f) + 0.3f * cosf((float)i * 0.071f);
    }
    for (size_t i = 0; i < batch; ++i) {
        labels[i] = (i * 5) % 11;
    }
}

typedef struct {
    const char *name;
    const Parameter *parameter;
} NamedParameter;

static size_t collect_parameters(const ViTModel *model, NamedParameter *out) {
    size_t count = 0;
    out[count++] = (NamedParameter){"patch_projection", &model->patch_projection.projection};
    out[count++] = (NamedParameter){"patch_bias", &model->patch_projection.bias};
    out[count++] = (NamedParameter){"cls_token", &model->token_embedding.cls_token};
    out[count++] = (NamedParameter){"positional", &model->token_embedding.positional_embeddings};
    for (size_t layer = 0; layer < model->config.layers; ++layer) {
        const ViTEncoderBlock *block = &model->blocks[layer];
        out[count++] = (NamedParameter){"query_key_value", &block->query_key_value};
        out[count++] = (NamedParameter){"query_key_value_bias", &block->query_key_value_bias};
        out[count++] = (NamedParameter){"attention_output", &block->attention_output};
        out[count++] = (NamedParameter){"attention_output_bias", &block->attention_output_bias};
        out[count++] = (NamedParameter){"mlp_input", &block->mlp_input};
        out[count++] = (NamedParameter){"mlp_input_bias", &block->mlp_input_bias};
        out[count++] = (NamedParameter){"mlp_output", &block->mlp_output};
        out[count++] = (NamedParameter){"mlp_output_bias", &block->mlp_output_bias};
        out[count++] = (NamedParameter){"attention_gamma", &block->attention_gamma};
        out[count++] = (NamedParameter){"attention_beta", &block->attention_beta};
        out[count++] = (NamedParameter){"mlp_gamma", &block->mlp_gamma};
        out[count++] = (NamedParameter){"mlp_beta", &block->mlp_beta};
    }
    out[count++] = (NamedParameter){"final_gamma", &model->final_gamma};
    out[count++] = (NamedParameter){"final_beta", &model->final_beta};
    out[count++] = (NamedParameter){"head", &model->classification_head.projection};
    out[count++] = (NamedParameter){"head_bias", &model->classification_head.bias};
    return count;
}

static int compute_gradients(ViTModel *model, ViTModelCache *cache,
                             const float *images, const size_t *labels,
                             float *image_grad, float *loss) {
    vit_model_zero_grad(model);
    return vit_model_forward(model, images, cache->batch, cache) != 0 ||
        ops_softmax_cross_entropy(&cache->logits, labels, loss,
                                  &cache->logits) != 0 ||
        vit_model_backward(model, images, cache, image_grad) != 0;
}

static int compare_gradients(const ViTModel *reference,
                             const ViTModel *accelerated) {
    NamedParameter reference_parameters[64];
    NamedParameter accelerated_parameters[64];
    const size_t count = collect_parameters(reference, reference_parameters);
    collect_parameters(accelerated, accelerated_parameters);
    int failures = 0;
    for (size_t p = 0; p < count; ++p) {
        const Tensor *expected = &reference_parameters[p].parameter->value;
        const Tensor *actual = &accelerated_parameters[p].parameter->value;
        float largest_gradient = 0.0f;
        float largest_error = 0.0f;
        for (size_t i = 0; i < tensor_numel(expected); ++i) {
            largest_gradient = fmaxf(largest_gradient, fabsf(expected->grad[i]));
            largest_error = fmaxf(largest_error,
                                  fabsf(expected->grad[i] - actual->grad[i]));
        }
        const float relative_error =
            largest_error / fmaxf(largest_gradient, 1e-6f);
        if (relative_error > 1e-3f || !isfinite(relative_error)) {
            fprintf(stderr, "FAIL: gradient %s relative error %.3e\n",
                    reference_parameters[p].name, relative_error);
            ++failures;
        }
    }
    return failures;
}

static int train_steps(ViTModel *model, ViTModelCache *cache,
                       const float *images, const size_t *labels,
                       size_t training_steps, float *losses) {
    float class_weights[11];
    for (size_t class_index = 0; class_index < 11; ++class_index) {
        class_weights[class_index] = 0.5f + 0.1f * (float)class_index;
    }
    for (size_t step = 0; step < training_steps; ++step) {
        const float *step_class_weights =
            step >= training_steps / 2 ? class_weights : NULL;
        if (vit_model_train_batch_smoothed(model, images, labels, cache->batch,
                                           cache,
                                           1e-3f, 0.05f, step_class_weights,
                                           0.1f, &losses[step]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int run_scenario(DeviceBackend *backend, const TestScenario *scenario) {
    float images[MAX_BATCH * IMAGE_SIZE];
    size_t labels[MAX_BATCH];
    float image_grad[MAX_BATCH * IMAGE_SIZE] = {0};
    fill_batch(images, labels, scenario->batch);
    printf("%s\n", scenario->name);

    ViTModel reference = {0};
    ViTModelCache reference_cache = {0};
    float reference_loss = 0.0f;
    float reference_losses[MAX_TRAINING_STEPS];
    int failures = vit_model_init(&reference, &scenario->config) != 0 ||
        vit_model_cache_init(&reference_cache, &reference, scenario->batch) != 0 ||
        compute_gradients(&reference, &reference_cache, images, labels,
                          image_grad, &reference_loss) != 0;

    if (device_backend_enable(backend) != 0) {
        fprintf(stderr, "FAIL: enabling " DEVICE_NAME " device execution\n");
        return 1;
    }
    ViTModel accelerated = {0};
    ViTModelCache accelerated_cache = {0};
    float accelerated_loss = 0.0f;
    float accelerated_losses[MAX_TRAINING_STEPS];
    failures += vit_model_init(&accelerated, &scenario->config) != 0 ||
        vit_model_cache_init(&accelerated_cache, &accelerated,
                             scenario->batch) != 0 ||
        compute_gradients(&accelerated, &accelerated_cache, images, labels,
                          image_grad, &accelerated_loss) != 0;
    if (failures == 0) {
        ops_synchronize();
        printf("  forward loss cpu=%.7f metal=%.7f\n", reference_loss,
               accelerated_loss);
        failures += fabsf(reference_loss - accelerated_loss) > 1e-4f;
        failures += compare_gradients(&reference, &accelerated);
        const float small_gradient_norm = 0.05f;
        failures += vit_model_clip_gradients(&reference,
                                             small_gradient_norm) != 0 ||
            vit_model_clip_gradients(&accelerated, small_gradient_norm) != 0;
        ops_synchronize();
        failures += compare_gradients(&reference, &accelerated);
    }

    failures += train_steps(&accelerated, &accelerated_cache, images, labels,
                            scenario->training_steps, accelerated_losses) != 0;
    device_backend_disable(backend);
    failures += train_steps(&reference, &reference_cache, images, labels,
                            scenario->training_steps, reference_losses) != 0;
    for (size_t step = 0; failures == 0 && step < scenario->training_steps;
         ++step) {
        printf("  step %2zu loss cpu=%.6f metal=%.6f\n", step + 1,
               reference_losses[step], accelerated_losses[step]);
        failures += fabsf(reference_losses[step] - accelerated_losses[step]) >
                    1e-3f;
    }

    vit_model_cache_free(&accelerated_cache);
    vit_model_free(&accelerated);
    vit_model_cache_free(&reference_cache);
    vit_model_free(&reference);
    return failures;
}

int main(void) {
    DeviceBackend *backend = NULL;
    if (device_backend_create(&backend) != 0 ||
        !device_backend_available(backend)) {
        fprintf(stderr, DEVICE_NAME " unavailable; skipping\n");
        return 0;
    }
    const TestScenario scenarios[] = {
        {"patch 4 (threadgroup-memory attention)",
         {1, 28, 28, 4, 32, 4, 2, 11}, 8, 12},
        {"patch 2 (long-sequence attention fallback)",
         {1, 28, 28, 2, 32, 4, 2, 11}, 2, 4},
        {"large head dimension (legacy attention fallback)",
         {1, 28, 28, 4, 128, 1, 1, 11}, 2, 3},
    };
    int failures = 0;
    for (size_t i = 0; i < sizeof(scenarios) / sizeof(*scenarios); ++i) {
        failures += run_scenario(backend, &scenarios[i]);
    }
    device_backend_free(backend);
    printf(DEVICE_NAME " device execution: %s\n",
           failures == 0 ? "ok" : "failed");
    return failures == 0 ? 0 : 1;
}

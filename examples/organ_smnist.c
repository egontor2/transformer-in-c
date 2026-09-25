#include "dataset.h"
#ifdef ENABLE_MPS
#include "mps_backend.h"
#include "ops.h"
#endif
#include "vit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long next_random(unsigned long *state) {
    *state = *state * 1664525UL + 1013904223UL;
    return *state;
}

static int shuffle_dataset(VisionDataset *dataset, unsigned long *state) {
    const size_t image_size = dataset->height * dataset->width *
                              dataset->channels;
    float *temporary_image = malloc(image_size * sizeof(*temporary_image));
    if (!temporary_image) {
        return -1;
    }
    for (size_t index = dataset->sample_count; index > 1; --index) {
        const size_t other = next_random(state) % index;
        if (other == index - 1) {
            continue;
        }
        memcpy(temporary_image,
               dataset->images + (index - 1) * image_size,
               image_size * sizeof(*temporary_image));
        memcpy(dataset->images + (index - 1) * image_size,
               dataset->images + other * image_size,
               image_size * sizeof(*temporary_image));
        memcpy(dataset->images + other * image_size, temporary_image,
               image_size * sizeof(*temporary_image));
        const size_t label = dataset->labels[index - 1];
        dataset->labels[index - 1] = dataset->labels[other];
        dataset->labels[other] = label;
    }
    free(temporary_image);
    return 0;
}

static int train_dataset(ViTModel *model, VisionDataset *dataset,
                         const VisionDataset *validation,
                         const char *checkpoint_path,
                         size_t batch_size, size_t epochs,
                         float learning_rate, float weight_decay) {
    ViTModelCache cache = {0};
    if (vit_model_cache_init(&cache, model, batch_size) != 0) {
        return -1;
    }
    unsigned long random_state = 42;
    float best_accuracy = -1.0f;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle_dataset(dataset, &random_state) != 0) {
            vit_model_cache_free(&cache);
            return -1;
        }
        float epoch_loss = 0.0f;
        size_t batches = 0;
        for (size_t offset = 0; offset < dataset->sample_count;
             offset += batch_size) {
            const size_t batch = dataset->sample_count - offset < batch_size
                ? dataset->sample_count - offset
                : batch_size;
            if (batch != batch_size) {
                break;
            }
            float loss = 0.0f;
            const size_t image_size = dataset->height * dataset->width *
                                      dataset->channels;
            if (vit_model_train_batch(
                    model, dataset->images + offset * image_size,
                    dataset->labels + offset, batch, &cache, learning_rate,
                    weight_decay, &loss) != 0) {
                vit_model_cache_free(&cache);
                return -1;
            }
            epoch_loss += loss;
            ++batches;
        }
        if (batches == 0) {
            vit_model_cache_free(&cache);
            return -1;
        }
        ViTEvaluationMetrics metrics = {0};
        if (vit_model_evaluate(model, validation->images, validation->labels,
                               validation->sample_count, &metrics) != 0) {
            vit_model_cache_free(&cache);
            return -1;
        }
        if (metrics.accuracy > best_accuracy) {
            best_accuracy = metrics.accuracy;
            if (vit_model_save(model, checkpoint_path) != 0) {
                vit_model_cache_free(&cache);
                return -1;
            }
        }
        printf("epoch %zu/%zu loss=%.6f val_accuracy=%.4f val_loss=%.6f\n",
               epoch + 1, epochs, epoch_loss / (float)batches,
               metrics.accuracy, metrics.average_loss);
    }
    vit_model_cache_free(&cache);
    return 0;
}

int main(int argc, char **argv) {
#ifdef ENABLE_MPS
    MPSBackend *mps_backend = NULL;
    if (mps_backend_create(&mps_backend) != 0 ||
        !mps_backend_available(mps_backend)) {
        fprintf(stderr, "MPS backend unavailable\n");
        mps_backend_free(mps_backend);
        return 1;
    }
    ops_set_gemm_backend(mps_backend_gemm_callback, mps_backend);
    ops_set_gemm_backward_backend(mps_backend_gemm_backward_callback,
                                  mps_backend);
#endif
    const char *train_manifest = argc > 1 ? argv[1] : "data/organ_smnist/train.csv";
    const char *validation_manifest =
        argc > 2 ? argv[2] : "data/organ_smnist/val.csv";
    const char *test_manifest =
        argc > 3 ? argv[3] : "data/organ_smnist/test.csv";
    const char *checkpoint_path =
        argc > 4 ? argv[4] : "build/organ_smnist.vit";
    const size_t batch_size = argc > 5 ? (size_t)strtoul(argv[5], NULL, 10) : 32;
    const size_t epochs = argc > 6 ? (size_t)strtoul(argv[6], NULL, 10) : 5;
    const float learning_rate = argc > 7 ? strtof(argv[7], NULL) : 0.0001f;
    const float weight_decay = argc > 8 ? strtof(argv[8], NULL) : 0.01f;

    VisionDataset train = {0};
    VisionDataset validation = {0};
    VisionDataset test = {0};
    if (vision_dataset_load_pgm_csv(train_manifest, &train) != 0 ||
        vision_dataset_load_pgm_csv(validation_manifest, &validation) != 0 ||
        vision_dataset_load_pgm_csv(test_manifest, &test) != 0 ||
        train.height != 28 || train.width != 28 || train.channels != 1 ||
        validation.height != train.height || validation.width != train.width ||
        validation.channels != train.channels ||
        test.height != train.height || test.width != train.width ||
        test.channels != train.channels) {
        fprintf(stderr, "failed to load 28x28 grayscale OrganSMNIST manifests\n");
#ifdef ENABLE_MPS
        ops_reset_gemm_backend();
        mps_backend_free(mps_backend);
#endif
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }

    const ViTConfig config = {
        .channels = 1,
        .height = 28,
        .width = 28,
        .patch_size = 4,
        .d_model = 64,
        .heads = 4,
        .layers = 2,
        .classes = 11,
    };
    ViTModel model = {0};
    if (vit_model_init(&model, &config) != 0 ||
        train_dataset(&model, &train, &validation, checkpoint_path, batch_size,
                      epochs, learning_rate, weight_decay) != 0) {
        fprintf(stderr, "OrganSMNIST training failed\n");
#ifdef ENABLE_MPS
        ops_reset_gemm_backend();
        mps_backend_free(mps_backend);
#endif
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }

    if (vit_model_load(&model, checkpoint_path) != 0) {
        fprintf(stderr, "failed to restore best validation checkpoint\n");
#ifdef ENABLE_MPS
        ops_reset_gemm_backend();
        mps_backend_free(mps_backend);
#endif
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }
    ViTEvaluationMetrics metrics = {0};
    if (vit_model_evaluate(&model, validation.images, validation.labels,
                           validation.sample_count, &metrics) != 0) {
        fprintf(stderr, "OrganSMNIST evaluation failed\n");
#ifdef ENABLE_MPS
        ops_reset_gemm_backend();
        mps_backend_free(mps_backend);
#endif
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }
    printf("validation accuracy=%.4f loss=%.6f samples/s=%.2f\n",
           metrics.accuracy, metrics.average_loss,
           metrics.samples_per_second);
    if (vit_model_evaluate(&model, test.images, test.labels,
                           test.sample_count, &metrics) != 0) {
        fprintf(stderr, "OrganSMNIST test evaluation failed\n");
#ifdef ENABLE_MPS
        ops_reset_gemm_backend();
        mps_backend_free(mps_backend);
#endif
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }
    printf("test accuracy=%.4f loss=%.6f samples/s=%.2f\n",
           metrics.accuracy, metrics.average_loss,
           metrics.samples_per_second);

    vit_model_free(&model);
    vision_dataset_free(&train);
    vision_dataset_free(&validation);
    vision_dataset_free(&test);
#ifdef ENABLE_MPS
    ops_reset_gemm_backend();
    mps_backend_free(mps_backend);
#endif
    return 0;
}

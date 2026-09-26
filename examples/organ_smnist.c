#include "dataset.h"
#ifdef ENABLE_MPS
#include "mps_backend.h"
#include "ops.h"
#endif
#include "vit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PI_F 3.14159265358979323846f

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

static int print_class_metrics(const ViTModel *model,
                               const VisionDataset *dataset) {
    size_t correct[11] = {0};
    size_t total[11] = {0};
    ViTModelCache cache = {0};
    if (vit_model_cache_init(&cache, model, 1) != 0) {
        return -1;
    }
    const size_t image_size = dataset->height * dataset->width *
                              dataset->channels;
    for (size_t sample = 0; sample < dataset->sample_count; ++sample) {
        if (vit_model_forward(model, dataset->images + sample * image_size, 1,
                              &cache) != 0) {
            vit_model_cache_free(&cache);
            return -1;
        }
        size_t prediction = 0;
        for (size_t class_index = 1; class_index < model->config.classes;
             ++class_index) {
            if (cache.logits.data[class_index] >
                cache.logits.data[prediction]) {
                prediction = class_index;
            }
        }
        const size_t label = dataset->labels[sample];
        if (label < model->config.classes) {
            ++total[label];
            if (prediction == label) {
                ++correct[label];
            }
        }
    }
    vit_model_cache_free(&cache);
    printf("test accuracy by class:\n");
    for (size_t class_index = 0; class_index < model->config.classes;
         ++class_index) {
        const float accuracy = total[class_index] == 0
            ? 0.0f
            : (float)correct[class_index] / (float)total[class_index];
        printf("  class %zu: %.4f (%zu/%zu)\n", class_index, accuracy,
               correct[class_index], total[class_index]);
    }
    return 0;
}

typedef struct {
    size_t epoch;
    float learning_rate;
    float best_validation_loss;
    size_t plateau_bad_epochs;
} TrainingState;

static int save_training_state(const char *checkpoint_path,
                               const TrainingState *state) {
    char path[4096];
    if (!checkpoint_path || !state ||
        snprintf(path, sizeof(path), "%s.state", checkpoint_path) >=
            (int)sizeof(path)) {
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        return -1;
    }
    const int result = fwrite(state, sizeof(*state), 1, file) == 1 ? 0 : -1;
    fclose(file);
    return result;
}

static int load_training_state(const char *checkpoint_path,
                               TrainingState *state) {
    char path[4096];
    if (!checkpoint_path || !state ||
        snprintf(path, sizeof(path), "%s.state", checkpoint_path) >=
            (int)sizeof(path)) {
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }
    const int result = fread(state, sizeof(*state), 1, file) == 1 ? 0 : -1;
    fclose(file);
    return result;
}

static int train_dataset(ViTModel *model, VisionDataset *dataset,
                         const VisionDataset *validation,
                         const char *checkpoint_path,
                         size_t batch_size, size_t epochs,
                         float learning_rate, float weight_decay,
                         size_t warmup_epochs, const char *scheduler,
                         size_t patience, float factor,
                         int has_initial_checkpoint,
                         size_t early_stopping_patience,
                         const char *metrics_path,
                         const float *class_weights,
                         const char *initial_checkpoint_path,
                         float resume_learning_rate) {
    ViTModelCache cache = {0};
    if (vit_model_cache_init(&cache, model, batch_size) != 0) {
        return -1;
    }
    unsigned long random_state = 42;
    float best_validation_loss = FLT_MAX;
    float scheduled_learning_rate = learning_rate;
    size_t plateau_bad_epochs = 0;
    size_t early_stopping_bad_epochs = 0;
    TrainingState initial_state = {0};
    if (has_initial_checkpoint &&
        load_training_state(initial_checkpoint_path, &initial_state) == 0) {
        scheduled_learning_rate = initial_state.learning_rate;
        best_validation_loss = initial_state.best_validation_loss;
        plateau_bad_epochs = initial_state.plateau_bad_epochs;
        printf("resumed training state epoch=%zu lr=%.8f\n",
               initial_state.epoch, scheduled_learning_rate);
    }
    if (has_initial_checkpoint && resume_learning_rate > 0.0f) {
        scheduled_learning_rate = resume_learning_rate;
        printf("overriding resumed learning rate with %.8f\n",
               scheduled_learning_rate);
    }
    FILE *metrics_file = metrics_path ? fopen(metrics_path, "w") : NULL;
    if (metrics_path && !metrics_file) {
        vit_model_cache_free(&cache);
        return -1;
    }
    if (metrics_file) {
        fprintf(metrics_file,
                "epoch,learning_rate,train_loss,val_accuracy,val_loss\n");
    }
    if (has_initial_checkpoint) {
        ViTEvaluationMetrics initial_metrics = {0};
        if (vit_model_evaluate(model, validation->images, validation->labels,
                               validation->sample_count,
                               &initial_metrics) != 0) {
            vit_model_cache_free(&cache);
            if (metrics_file) fclose(metrics_file);
            return -1;
        }
        best_validation_loss = initial_metrics.average_loss;
        printf("resumed checkpoint validation accuracy=%.4f val_loss=%.6f\n",
               initial_metrics.accuracy, initial_metrics.average_loss);
    }
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_learning_rate = scheduled_learning_rate;
        if (!has_initial_checkpoint && warmup_epochs > 0 &&
            epoch < warmup_epochs) {
            epoch_learning_rate =
                learning_rate * (float)(epoch + 1) / (float)warmup_epochs;
        } else if (strcmp(scheduler, "cosine") == 0 &&
                   epochs > warmup_epochs + 1) {
            const size_t decay_steps = epochs - warmup_epochs - 1;
            const size_t decay_step = epoch > warmup_epochs
                ? epoch - warmup_epochs
                : 0;
            const float progress = (float)decay_step / (float)decay_steps;
            epoch_learning_rate = scheduled_learning_rate *
                (0.5f * (1.0f + cosf(PI_F * progress)));
        }
        if (shuffle_dataset(dataset, &random_state) != 0) {
            vit_model_cache_free(&cache);
            if (metrics_file) fclose(metrics_file);
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
            if (vit_model_train_batch_weighted(
                    model, dataset->images + offset * image_size,
                    dataset->labels + offset, batch, &cache,
                    epoch_learning_rate,
                    weight_decay, class_weights, &loss) != 0) {
                vit_model_cache_free(&cache);
                if (metrics_file) fclose(metrics_file);
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
        const int validation_loss_improved =
            metrics.average_loss < best_validation_loss - 1e-5f;
        if (validation_loss_improved) {
            if (vit_model_save(model, checkpoint_path) != 0) {
                vit_model_cache_free(&cache);
                return -1;
            }
            TrainingState state = {
                .epoch = epoch + 1,
                .learning_rate = scheduled_learning_rate,
                .best_validation_loss = metrics.average_loss,
                .plateau_bad_epochs = 0,
            };
            if (save_training_state(checkpoint_path, &state) != 0) {
                vit_model_cache_free(&cache);
                return -1;
            }
        }
        if (strcmp(scheduler, "plateau") == 0) {
            if (validation_loss_improved) {
                best_validation_loss = metrics.average_loss;
                plateau_bad_epochs = 0;
            } else {
                ++plateau_bad_epochs;
                if (patience > 0 && plateau_bad_epochs >= patience) {
                    scheduled_learning_rate *= factor;
                    if (scheduled_learning_rate < 1e-7f) {
                        scheduled_learning_rate = 1e-7f;
                    }
                    plateau_bad_epochs = 0;
                }
            }
        }
        if (validation_loss_improved) {
            best_validation_loss = metrics.average_loss;
            early_stopping_bad_epochs = 0;
        } else {
            ++early_stopping_bad_epochs;
        }
        printf("epoch %zu/%zu lr=%.8f loss=%.6f val_accuracy=%.4f val_loss=%.6f\n",
               epoch + 1, epochs, epoch_learning_rate,
               epoch_loss / (float)batches, metrics.accuracy,
               metrics.average_loss);
        if (metrics_file) {
            fprintf(metrics_file, "%zu,%.9g,%.9g,%.9g,%.9g\n", epoch + 1,
                    epoch_learning_rate, epoch_loss / (float)batches,
                    metrics.accuracy, metrics.average_loss);
            fflush(metrics_file);
        }
        if (early_stopping_patience > 0 &&
            early_stopping_bad_epochs >= early_stopping_patience) {
            printf("early stopping after %zu epochs without validation loss improvement\n",
                   early_stopping_bad_epochs);
            break;
        }
    }
    vit_model_cache_free(&cache);
    if (metrics_file) fclose(metrics_file);
    return 0;
}

static void print_usage(const char *program) {
    printf("Usage: %s [TRAIN VAL TEST CHECKPOINT BATCH EPOCHS LR DECAY "
           "WARMUP SCHEDULER PATIENCE FACTOR [RESUME]]\n", program);
    printf("   or: %s [options]\n\n", program);
    printf("Options: --train PATH --val PATH --test PATH --checkpoint PATH\n");
    printf("         --batch N --epochs N --lr VALUE --weight-decay VALUE\n");
    printf("         --warmup N --scheduler plateau|cosine|constant\n");
    printf("         --patience N --factor VALUE --resume PATH\n");
    printf("         --early-stopping N (0 disables)\n");
    printf("         --metrics-csv PATH\n");
    printf("         --class-weights balanced\n");
    printf("         --resume-lr VALUE (override saved LR)\n");
}

static int parse_named_arguments(int argc, char **argv,
                                 const char **train_manifest,
                                 const char **validation_manifest,
                                 const char **test_manifest,
                                 const char **checkpoint_path,
                                 size_t *batch_size, size_t *epochs,
                                 float *learning_rate, float *weight_decay,
                                 size_t *warmup_epochs, const char **scheduler,
                                 size_t *patience, float *factor,
                                 const char **resume_path,
                                 size_t *early_stopping_patience,
                                 const char **metrics_path,
                                 int *balanced_class_weights,
                                 float *resume_learning_rate) {
    for (int index = 1; index < argc; ++index) {
        const char *option = argv[index];
        if (strcmp(option, "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", option);
            return -1;
        }
        const char *value = argv[++index];
        if (strcmp(option, "--train") == 0) *train_manifest = value;
        else if (strcmp(option, "--val") == 0) *validation_manifest = value;
        else if (strcmp(option, "--test") == 0) *test_manifest = value;
        else if (strcmp(option, "--checkpoint") == 0) *checkpoint_path = value;
        else if (strcmp(option, "--batch") == 0)
            *batch_size = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--epochs") == 0)
            *epochs = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--lr") == 0) *learning_rate = strtof(value, NULL);
        else if (strcmp(option, "--weight-decay") == 0)
            *weight_decay = strtof(value, NULL);
        else if (strcmp(option, "--warmup") == 0)
            *warmup_epochs = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--scheduler") == 0) *scheduler = value;
        else if (strcmp(option, "--patience") == 0)
            *patience = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--factor") == 0) *factor = strtof(value, NULL);
        else if (strcmp(option, "--resume") == 0) *resume_path = value;
        else if (strcmp(option, "--early-stopping") == 0)
            *early_stopping_patience = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--metrics-csv") == 0) *metrics_path = value;
        else if (strcmp(option, "--class-weights") == 0) {
            if (strcmp(value, "balanced") != 0) {
                fprintf(stderr, "--class-weights accepts only balanced\n");
                return -1;
            }
            *balanced_class_weights = 1;
        }
        else if (strcmp(option, "--resume-lr") == 0)
            *resume_learning_rate = strtof(value, NULL);
        else {
            fprintf(stderr, "unknown option: %s\n", option);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *train_manifest = "data/organ_smnist/train.csv";
    const char *validation_manifest = "data/organ_smnist/val.csv";
    const char *test_manifest = "data/organ_smnist/test.csv";
    const char *checkpoint_path = "build/organ_smnist.vit";
    size_t batch_size = 32;
    size_t epochs = 5;
    float learning_rate = 0.0001f;
    float weight_decay = 0.01f;
    size_t warmup_epochs = 1;
    const char *scheduler = "plateau";
    size_t patience = 2;
    float factor = 0.5f;
    const char *resume_path = NULL;
    size_t early_stopping_patience = 0;
    const char *metrics_path = NULL;
    int balanced_class_weights = 0;
    float resume_learning_rate = 0.0f;
    if (argc > 1 && argv[1][0] == '-') {
        const int parse_result = parse_named_arguments(
            argc, argv, &train_manifest, &validation_manifest, &test_manifest,
            &checkpoint_path, &batch_size, &epochs, &learning_rate,
            &weight_decay, &warmup_epochs, &scheduler, &patience, &factor,
            &resume_path, &early_stopping_patience, &metrics_path,
            &balanced_class_weights, &resume_learning_rate);
        if (parse_result != 0) {
            return parse_result > 0 ? 0 : 1;
        }
    } else {
        train_manifest = argc > 1 ? argv[1] : train_manifest;
        validation_manifest = argc > 2 ? argv[2] : validation_manifest;
        test_manifest = argc > 3 ? argv[3] : test_manifest;
        checkpoint_path = argc > 4 ? argv[4] : checkpoint_path;
        batch_size = argc > 5 ? (size_t)strtoul(argv[5], NULL, 10) : batch_size;
        epochs = argc > 6 ? (size_t)strtoul(argv[6], NULL, 10) : epochs;
        learning_rate = argc > 7 ? strtof(argv[7], NULL) : learning_rate;
        weight_decay = argc > 8 ? strtof(argv[8], NULL) : weight_decay;
        warmup_epochs = argc > 9 ? (size_t)strtoul(argv[9], NULL, 10) : warmup_epochs;
        scheduler = argc > 10 ? argv[10] : scheduler;
        patience = argc > 11 ? (size_t)strtoul(argv[11], NULL, 10) : patience;
        factor = argc > 12 ? strtof(argv[12], NULL) : factor;
        resume_path = argc > 13 ? argv[13] : resume_path;
        early_stopping_patience = argc > 14
            ? (size_t)strtoul(argv[14], NULL, 10)
            : early_stopping_patience;
        metrics_path = argc > 15 ? argv[15] : metrics_path;
    }
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
    if ((strcmp(scheduler, "plateau") != 0 &&
         strcmp(scheduler, "cosine") != 0 &&
         strcmp(scheduler, "constant") != 0) ||
        patience == 0 || factor <= 0.0f || factor >= 1.0f) {
        fprintf(stderr, "scheduler must be plateau, cosine, or constant; "
                        "patience > 0 and factor in (0,1)\n");
        return 1;
    }

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
    float class_weights[11];
    const float *class_weights_ptr = NULL;
    if (balanced_class_weights) {
        size_t counts[11] = {0};
        for (size_t i = 0; i < train.sample_count; ++i) {
            if (train.labels[i] < 11) {
                ++counts[train.labels[i]];
            }
        }
        for (size_t class_index = 0; class_index < 11; ++class_index) {
            class_weights[class_index] = counts[class_index] == 0
                ? 0.0f
                : (float)train.sample_count /
                  (11.0f * (float)counts[class_index]);
        }
        class_weights_ptr = class_weights;
    }
    ViTModel model = {0};
    if (vit_model_init(&model, &config) != 0 ||
        (resume_path && vit_model_load(&model, resume_path) != 0) ||
        train_dataset(&model, &train, &validation, checkpoint_path, batch_size,
                  epochs, learning_rate, weight_decay, warmup_epochs,
                  scheduler, patience, factor, resume_path != NULL,
                  early_stopping_patience, metrics_path,
                  class_weights_ptr, resume_path,
                  resume_learning_rate) != 0) {
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
    if (print_class_metrics(&model, &test) != 0) {
        fprintf(stderr, "OrganSMNIST per-class evaluation failed\n");
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }

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

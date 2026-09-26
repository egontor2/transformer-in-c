#include "augmentation.h"
#include "dataset.h"
#if defined(ENABLE_MPS)
#include "mps_backend.h"
#include "ops.h"
#elif defined(ENABLE_CUDA)
#include "cuda_backend.h"
#endif
#include "vit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PI_F 3.14159265358979323846f

#if defined(ENABLE_MPS)
typedef MPSBackend Accelerator;
#elif defined(ENABLE_CUDA)
typedef CudaBackend Accelerator;
#else
typedef struct Accelerator Accelerator;
#endif

static void stop_accelerator(Accelerator *accelerator) {
#if defined(ENABLE_MPS)
    ops_reset_gemm_backend();
    mps_backend_free(accelerator);
#elif defined(ENABLE_CUDA)
    cuda_backend_free(accelerator);
#else
    (void)accelerator;
#endif
}

static int start_accelerator(Accelerator **accelerator) {
    *accelerator = NULL;
#if defined(ENABLE_MPS)
    if (mps_backend_create(accelerator) != 0 ||
        !mps_backend_available(*accelerator)) {
        fprintf(stderr, "MPS backend unavailable\n");
        mps_backend_free(*accelerator);
        *accelerator = NULL;
        return -1;
    }
    ops_set_gemm_backend(mps_backend_gemm_callback, *accelerator);
    ops_set_gemm_backward_backend(mps_backend_gemm_backward_callback,
                                  *accelerator);
    if (mps_backend_enable_device_execution(*accelerator) != 0) {
        fprintf(stderr, "Metal device execution unavailable\n");
        stop_accelerator(*accelerator);
        *accelerator = NULL;
        return -1;
    }
#elif defined(ENABLE_CUDA)
    if (cuda_backend_create(accelerator) != 0 ||
        !cuda_backend_available(*accelerator) ||
        cuda_backend_enable_device_execution(*accelerator) != 0) {
        fprintf(stderr, "CUDA backend unavailable\n");
        cuda_backend_free(*accelerator);
        *accelerator = NULL;
        return -1;
    }
#endif
    return 0;
}

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

typedef struct {
    float mean;
    float inverse_standard_deviation;
} PixelStatistics;

static PixelStatistics compute_pixel_statistics(const VisionDataset *dataset) {
    const size_t pixel_count = dataset->sample_count * dataset->height *
                               dataset->width * dataset->channels;
    double sum = 0.0;
    double squared_sum = 0.0;
    for (size_t i = 0; i < pixel_count; ++i) {
        sum += dataset->images[i];
        squared_sum += (double)dataset->images[i] * dataset->images[i];
    }
    const double mean = sum / (double)pixel_count;
    const double variance = squared_sum / (double)pixel_count - mean * mean;
    const double minimum_variance = 1e-12;
    PixelStatistics statistics = {
        .mean = (float)mean,
        .inverse_standard_deviation = (float)(1.0 / sqrt(
            variance > minimum_variance ? variance : minimum_variance)),
    };
    return statistics;
}

static float normalize_pixel(float pixel, const PixelStatistics *statistics) {
    return (pixel - statistics->mean) * statistics->inverse_standard_deviation;
}

static void normalize_dataset(VisionDataset *dataset,
                              const PixelStatistics *statistics) {
    const size_t pixel_count = dataset->sample_count * dataset->height *
                               dataset->width * dataset->channels;
    for (size_t i = 0; i < pixel_count; ++i) {
        dataset->images[i] = normalize_pixel(dataset->images[i], statistics);
    }
}

static DataAugmentation standard_pixel_augmentation(void) {
    DataAugmentation augmentation = {
        .max_shift_pixels = 2.0f,
        .max_rotation_degrees = 10.0f,
        .max_scale_delta = 0.1f,
        .max_brightness_delta = 0.05f,
        .max_contrast_delta = 0.1f,
        .noise_standard_deviation = 0.01f,
        .erasing_probability = 0.25f,
        .erasing_max_side_fraction = 0.4f,
    };
    return augmentation;
}

static DataAugmentation normalize_augmentation(
    const DataAugmentation *pixel_augmentation,
    const PixelStatistics *statistics) {
    DataAugmentation augmentation = *pixel_augmentation;
    augmentation.max_brightness_delta *= statistics->inverse_standard_deviation;
    augmentation.noise_standard_deviation *=
        statistics->inverse_standard_deviation;
    augmentation.erasing_value = normalize_pixel(statistics->mean, statistics);
    augmentation.minimum_value = normalize_pixel(0.0f, statistics);
    augmentation.maximum_value = normalize_pixel(1.0f, statistics);
    return augmentation;
}

static void print_augmentation(const DataAugmentation *pixel_augmentation,
                               float label_smoothing) {
    printf("augmentation shift=%.2f rotation=%.2f scale=%.2f brightness=%.3f "
           "contrast=%.2f noise=%.3f erasing=%.2f/%.2f label_smoothing=%.2f\n",
           pixel_augmentation->max_shift_pixels,
           pixel_augmentation->max_rotation_degrees,
           pixel_augmentation->max_scale_delta,
           pixel_augmentation->max_brightness_delta,
           pixel_augmentation->max_contrast_delta,
           pixel_augmentation->noise_standard_deviation,
           pixel_augmentation->erasing_probability,
           pixel_augmentation->erasing_max_side_fraction, label_smoothing);
}

static size_t predicted_class(const float *probabilities, size_t classes) {
    size_t prediction = 0;
    for (size_t class_index = 1; class_index < classes; ++class_index) {
        if (probabilities[class_index] > probabilities[prediction]) {
            prediction = class_index;
        }
    }
    return prediction;
}

static int print_prediction_metrics(const char *split_name,
                                    const float *probabilities,
                                    const size_t *labels, size_t sample_count,
                                    size_t classes, int print_details) {
    size_t *total = calloc(classes, sizeof(*total));
    size_t *predicted = calloc(classes, sizeof(*predicted));
    size_t *confusion = calloc(classes * classes, sizeof(*confusion));
    if (!total || !predicted || !confusion) {
        free(total);
        free(predicted);
        free(confusion);
        return -1;
    }
    const double minimum_probability = 1e-12;
    double negative_log_likelihood = 0.0;
    size_t correct = 0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const float *sample_probabilities = probabilities + sample * classes;
        const size_t label = labels[sample];
        const size_t prediction = predicted_class(sample_probabilities, classes);
        if (label < classes) {
            ++total[label];
            ++predicted[prediction];
            ++confusion[label * classes + prediction];
            correct += prediction == label;
            negative_log_likelihood -= log(fmax(
                (double)sample_probabilities[label], minimum_probability));
        }
    }
    double balanced_accuracy = 0.0;
    double macro_f1 = 0.0;
    for (size_t class_index = 0; class_index < classes; ++class_index) {
        const size_t true_positive = confusion[class_index * classes +
                                                class_index];
        const double recall = total[class_index] == 0
            ? 0.0
            : (double)true_positive / (double)total[class_index];
        const double precision = predicted[class_index] == 0
            ? 0.0
            : (double)true_positive / (double)predicted[class_index];
        balanced_accuracy += recall;
        macro_f1 += precision + recall == 0.0
            ? 0.0
            : 2.0 * precision * recall / (precision + recall);
    }
    balanced_accuracy /= (double)classes;
    macro_f1 /= (double)classes;
    printf("%s accuracy=%.4f loss=%.6f balanced_accuracy=%.4f macro_f1=%.4f\n",
           split_name, (double)correct / (double)sample_count,
           negative_log_likelihood / (double)sample_count, balanced_accuracy,
           macro_f1);
    if (print_details) {
        printf("%s accuracy by class:\n", split_name);
        for (size_t class_index = 0; class_index < classes; ++class_index) {
            const size_t true_positive =
                confusion[class_index * classes + class_index];
            printf("  class %zu: %.4f (%zu/%zu)\n", class_index,
                   total[class_index] == 0
                       ? 0.0
                       : (double)true_positive / (double)total[class_index],
                   true_positive, total[class_index]);
        }
        printf("%s confusion matrix (rows=label, columns=prediction):\n",
               split_name);
        for (size_t row = 0; row < classes; ++row) {
            printf("  %zu:", row);
            for (size_t col = 0; col < classes; ++col) {
                printf(" %zu", confusion[row * classes + col]);
            }
            putchar('\n');
        }
    }
    free(total);
    free(predicted);
    free(confusion);
    return 0;
}

static int print_class_metrics(const ViTModel *model,
                               const VisionDataset *dataset) {
    const size_t classes = model->config.classes;
    float *probabilities =
        malloc(dataset->sample_count * classes * sizeof(*probabilities));
    const int result = !probabilities ||
        vit_model_predict_probabilities(model, dataset->images,
                                        dataset->sample_count,
                                        probabilities) != 0 ||
        print_prediction_metrics("test", probabilities, dataset->labels,
                                 dataset->sample_count, classes, 1) != 0;
    free(probabilities);
    return result ? -1 : 0;
}

typedef struct {
    float shift_y;
    float shift_x;
} TestTimeShift;

static size_t test_time_shifts(size_t tta_views, TestTimeShift *shifts) {
    const TestTimeShift identity_and_neighbors[] = {
        {0.0f, 0.0f}, {-1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, -1.0f},
        {0.0f, 1.0f}, {-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f},
        {1.0f, 1.0f},
    };
    const size_t available = sizeof(identity_and_neighbors) /
                             sizeof(*identity_and_neighbors);
    if (tta_views != 1 && tta_views != 5 && tta_views != available) {
        return 0;
    }
    for (size_t i = 0; i < tta_views; ++i) {
        shifts[i] = identity_and_neighbors[i];
    }
    return tta_views;
}

static int accumulate_test_time_probabilities(const ViTModel *model,
                                              const VisionDataset *dataset,
                                              const TestTimeShift *shifts,
                                              size_t shift_count,
                                              float *shifted_images,
                                              float *view_probabilities,
                                              float *model_probabilities) {
    const size_t classes = model->config.classes;
    const size_t probability_count = dataset->sample_count * classes;
    memset(model_probabilities, 0, probability_count * sizeof(float));
    for (size_t view = 0; view < shift_count; ++view) {
        const int identity = shifts[view].shift_y == 0.0f &&
                             shifts[view].shift_x == 0.0f;
        const float *images = dataset->images;
        if (!identity) {
            if (data_augmentation_translate(
                    dataset->images, shifted_images, dataset->sample_count,
                    dataset->channels, dataset->height, dataset->width,
                    shifts[view].shift_y, shifts[view].shift_x) != 0) {
                return -1;
            }
            images = shifted_images;
        }
        if (vit_model_predict_probabilities(model, images,
                                            dataset->sample_count,
                                            view_probabilities) != 0) {
            return -1;
        }
        for (size_t i = 0; i < probability_count; ++i) {
            model_probabilities[i] += view_probabilities[i] / (float)shift_count;
        }
    }
    return 0;
}

typedef struct {
    const VisionDataset *dataset;
    const char *name;
    float *shifted_images;
    float *view_probabilities;
    float *model_probabilities;
    float *ensemble_probabilities;
} EvaluationSplit;

static int init_evaluation_split(EvaluationSplit *split,
                                 const VisionDataset *dataset,
                                 const char *name, size_t classes) {
    const size_t image_count = dataset->sample_count * dataset->channels *
                               dataset->height * dataset->width;
    const size_t probability_count = dataset->sample_count * classes;
    split->dataset = dataset;
    split->name = name;
    split->shifted_images = malloc(image_count * sizeof(float));
    split->view_probabilities = malloc(probability_count * sizeof(float));
    split->model_probabilities = malloc(probability_count * sizeof(float));
    split->ensemble_probabilities = calloc(probability_count, sizeof(float));
    return split->shifted_images && split->view_probabilities &&
                   split->model_probabilities && split->ensemble_probabilities
        ? 0
        : -1;
}

static void free_evaluation_split(EvaluationSplit *split) {
    free(split->shifted_images);
    free(split->view_probabilities);
    free(split->model_probabilities);
    free(split->ensemble_probabilities);
}

static int evaluate_ensemble(const char *checkpoint_list, size_t tta_views,
                             const VisionDataset *validation,
                             const VisionDataset *test) {
    const size_t classes = 11;
    TestTimeShift shifts[9];
    const size_t shift_count = test_time_shifts(tta_views, shifts);
    char *paths = malloc(strlen(checkpoint_list) + 1);
    EvaluationSplit splits[2] = {{0}, {0}};
    int result = shift_count == 0 || !paths ||
        init_evaluation_split(&splits[0], validation, "validation", classes) ||
        init_evaluation_split(&splits[1], test, "test", classes);
    if (shift_count == 0) {
        fprintf(stderr, "--tta accepts 1, 5 or 9 views\n");
    }
    size_t model_count = 0;
    if (result == 0) {
        strcpy(paths, checkpoint_list);
        printf("ensemble test-time views=%zu\n", shift_count);
    }
    for (char *path = result == 0 ? strtok(paths, ",") : NULL;
         result == 0 && path; path = strtok(NULL, ",")) {
        ViTConfig config = {0};
        ViTModel model = {0};
        result = vit_model_read_config(path, &config) != 0 ||
                 config.channels != validation->channels ||
                 config.height != validation->height ||
                 config.width != validation->width ||
                 config.classes != classes ||
                 vit_model_init(&model, &config) != 0 ||
                 vit_model_load(&model, path) != 0;
        if (result != 0) {
            fprintf(stderr, "failed to load ensemble checkpoint %s\n", path);
        }
        printf("model %s patch=%zu d_model=%zu heads=%zu layers=%zu\n", path,
               config.patch_size, config.d_model, config.heads, config.layers);
        for (size_t s = 0; result == 0 && s < 2; ++s) {
            EvaluationSplit *split = &splits[s];
            const size_t probability_count =
                split->dataset->sample_count * classes;
            result = accumulate_test_time_probabilities(
                &model, split->dataset, shifts, shift_count,
                split->shifted_images, split->view_probabilities,
                split->model_probabilities);
            for (size_t i = 0; result == 0 && i < probability_count; ++i) {
                split->ensemble_probabilities[i] +=
                    split->model_probabilities[i];
            }
            printf("  ");
            result = result ||
                print_prediction_metrics(split->name,
                                         split->model_probabilities,
                                         split->dataset->labels,
                                         split->dataset->sample_count, classes,
                                         0);
        }
        vit_model_free(&model);
        model_count += result == 0;
    }
    if (result == 0 && model_count > 0) {
        printf("ensemble of %zu models:\n", model_count);
        for (size_t s = 0; result == 0 && s < 2; ++s) {
            EvaluationSplit *split = &splits[s];
            const size_t probability_count =
                split->dataset->sample_count * classes;
            for (size_t i = 0; i < probability_count; ++i) {
                split->ensemble_probabilities[i] /= (float)model_count;
            }
            result = print_prediction_metrics(
                split->name, split->ensemble_probabilities,
                split->dataset->labels, split->dataset->sample_count, classes,
                s == 1);
        }
    }
    free_evaluation_split(&splits[0]);
    free_evaluation_split(&splits[1]);
    free(paths);
    return result == 0 && model_count > 0 ? 0 : -1;
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
                         float resume_learning_rate,
                         const DataAugmentation *augmentation,
                         float label_smoothing, uint64_t seed) {
    ViTModelCache cache = {0};
    if (vit_model_cache_init(&cache, model, batch_size) != 0) {
        return -1;
    }
    const float minimum_learning_rate = 1e-7f;
    unsigned long random_state = (unsigned long)seed;
    uint64_t augmentation_random_state = seed ^ 0x9E3779B97F4A7C15ULL;
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
            epoch_learning_rate = fmaxf(
                minimum_learning_rate,
                scheduled_learning_rate *
                    (0.5f * (1.0f + cosf(PI_F * progress))));
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
            float *augmented_images = NULL;
            const float *batch_images =
                dataset->images + offset * image_size;
            if (data_augmentation_enabled(augmentation)) {
                augmented_images = malloc(batch * image_size *
                                           sizeof(*augmented_images));
                if (!augmented_images ||
                    data_augmentation_apply(
                        augmentation, &augmentation_random_state,
                        batch_images, augmented_images, batch,
                        dataset->channels, dataset->height,
                        dataset->width) != 0) {
                    free(augmented_images);
                    vit_model_cache_free(&cache);
                    if (metrics_file) fclose(metrics_file);
                    return -1;
                }
                batch_images = augmented_images;
            }
            if (vit_model_train_batch_smoothed(
                    model, batch_images,
                    dataset->labels + offset, batch, &cache,
                    epoch_learning_rate,
                    weight_decay, class_weights, label_smoothing,
                    &loss) != 0) {
                free(augmented_images);
                vit_model_cache_free(&cache);
                if (metrics_file) fclose(metrics_file);
                return -1;
            }
            free(augmented_images);
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
                    if (scheduled_learning_rate < minimum_learning_rate) {
                        scheduled_learning_rate = minimum_learning_rate;
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
    printf("         --augment standard (lateral-safe augmentation preset)\n");
    printf("         --max-shift PIXELS --max-rotation DEGREES --max-scale DELTA\n");
    printf("         --brightness DELTA --contrast DELTA --noise-std VALUE\n");
    printf("         --erasing-prob P --erasing-size FRACTION\n");
    printf("         --label-smoothing VALUE\n");
    printf("         --seed N\n");
    printf("         --ensemble CKPT[,CKPT...] (evaluate only, no training)\n");
    printf("         --tta 1|5|9 (test-time shifted views, default 1)\n");
    printf("         --patch-size N --d-model N --heads N --layers N\n");
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
                                 float *resume_learning_rate,
                                 DataAugmentation *pixel_augmentation,
                                 float *label_smoothing,
                                 const char **ensemble_checkpoints,
                                 size_t *tta_views,
                                 size_t *patch_size, size_t *d_model,
                                 size_t *heads, size_t *layers,
                                 uint64_t *seed) {
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
        else if (strcmp(option, "--augment") == 0) {
            if (strcmp(value, "standard") != 0) {
                fprintf(stderr, "--augment accepts only standard\n");
                return -1;
            }
            *pixel_augmentation = standard_pixel_augmentation();
        }
        else if (strcmp(option, "--max-shift") == 0)
            pixel_augmentation->max_shift_pixels = strtof(value, NULL);
        else if (strcmp(option, "--max-rotation") == 0)
            pixel_augmentation->max_rotation_degrees = strtof(value, NULL);
        else if (strcmp(option, "--max-scale") == 0)
            pixel_augmentation->max_scale_delta = strtof(value, NULL);
        else if (strcmp(option, "--brightness") == 0)
            pixel_augmentation->max_brightness_delta = strtof(value, NULL);
        else if (strcmp(option, "--contrast") == 0)
            pixel_augmentation->max_contrast_delta = strtof(value, NULL);
        else if (strcmp(option, "--noise-std") == 0)
            pixel_augmentation->noise_standard_deviation = strtof(value, NULL);
        else if (strcmp(option, "--erasing-prob") == 0)
            pixel_augmentation->erasing_probability = strtof(value, NULL);
        else if (strcmp(option, "--erasing-size") == 0)
            pixel_augmentation->erasing_max_side_fraction = strtof(value, NULL);
        else if (strcmp(option, "--label-smoothing") == 0)
            *label_smoothing = strtof(value, NULL);
        else if (strcmp(option, "--seed") == 0)
            *seed = (uint64_t)strtoull(value, NULL, 10);
        else if (strcmp(option, "--ensemble") == 0)
            *ensemble_checkpoints = value;
        else if (strcmp(option, "--tta") == 0)
            *tta_views = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--patch-size") == 0)
            *patch_size = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--d-model") == 0)
            *d_model = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--heads") == 0)
            *heads = (size_t)strtoul(value, NULL, 10);
        else if (strcmp(option, "--layers") == 0)
            *layers = (size_t)strtoul(value, NULL, 10);
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
    DataAugmentation pixel_augmentation = {0};
    float label_smoothing = 0.0f;
    const char *ensemble_checkpoints = NULL;
    size_t tta_views = 1;
    size_t patch_size = 4;
    size_t d_model = 64;
    size_t heads = 4;
    size_t layers = 2;
    uint64_t seed = 42;
    if (argc > 1 && argv[1][0] == '-') {
        const int parse_result = parse_named_arguments(
            argc, argv, &train_manifest, &validation_manifest, &test_manifest,
            &checkpoint_path, &batch_size, &epochs, &learning_rate,
            &weight_decay, &warmup_epochs, &scheduler, &patience, &factor,
            &resume_path, &early_stopping_patience, &metrics_path,
            &balanced_class_weights, &resume_learning_rate, &pixel_augmentation,
            &label_smoothing, &ensemble_checkpoints, &tta_views,
            &patch_size, &d_model, &heads, &layers, &seed);
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
    Accelerator *accelerator = NULL;
    if (start_accelerator(&accelerator) != 0) {
        return 1;
    }
    if ((strcmp(scheduler, "plateau") != 0 &&
         strcmp(scheduler, "cosine") != 0 &&
         strcmp(scheduler, "constant") != 0) ||
        patience == 0 || factor <= 0.0f || factor >= 1.0f) {
        fprintf(stderr, "scheduler must be plateau, cosine, or constant; "
                        "patience > 0 and factor in (0,1)\n");
        return 1;
    }
    if (label_smoothing < 0.0f || label_smoothing >= 1.0f) {
        fprintf(stderr, "label smoothing must be in [0,1)\n");
        return 1;
    }
    const float default_erasing_side_fraction = 0.4f;
    if (pixel_augmentation.erasing_probability > 0.0f &&
        pixel_augmentation.erasing_max_side_fraction == 0.0f) {
        pixel_augmentation.erasing_max_side_fraction =
            default_erasing_side_fraction;
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
        stop_accelerator(accelerator);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }

    const ViTConfig config = {
        .channels = 1,
        .height = 28,
        .width = 28,
        .patch_size = patch_size,
        .d_model = d_model,
        .heads = heads,
        .layers = layers,
        .classes = 11,
    };
    if (patch_size == 0 || 28 % patch_size != 0 || d_model == 0 ||
        heads == 0 || d_model % heads != 0 || layers == 0) {
        fprintf(stderr, "invalid ViT configuration: patch-size must divide 28, "
                        "d-model must be divisible by heads, and values > 0\n");
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }
    const PixelStatistics pixel_statistics = compute_pixel_statistics(&train);
    normalize_dataset(&train, &pixel_statistics);
    normalize_dataset(&validation, &pixel_statistics);
    normalize_dataset(&test, &pixel_statistics);
    printf("pixel normalization mean=%.6f std=%.6f\n", pixel_statistics.mean,
           1.0f / pixel_statistics.inverse_standard_deviation);
    if (ensemble_checkpoints) {
        const int ensemble_result = evaluate_ensemble(
            ensemble_checkpoints, tta_views, &validation, &test);
        stop_accelerator(accelerator);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return ensemble_result == 0 ? 0 : 1;
    }
    const DataAugmentation augmentation =
        normalize_augmentation(&pixel_augmentation, &pixel_statistics);
    print_augmentation(&pixel_augmentation, label_smoothing);
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
    if (vit_model_init_seeded(&model, &config, seed) != 0 ||
        (resume_path && vit_model_load(&model, resume_path) != 0) ||
        train_dataset(&model, &train, &validation, checkpoint_path, batch_size,
                  epochs, learning_rate, weight_decay, warmup_epochs,
                  scheduler, patience, factor, resume_path != NULL,
                  early_stopping_patience, metrics_path,
                  class_weights_ptr, resume_path,
                  resume_learning_rate, &augmentation,
                  label_smoothing, seed) != 0) {
        fprintf(stderr, "OrganSMNIST training failed\n");
        stop_accelerator(accelerator);
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }

    if (vit_model_load(&model, checkpoint_path) != 0) {
        fprintf(stderr, "failed to restore best validation checkpoint\n");
        stop_accelerator(accelerator);
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
        stop_accelerator(accelerator);
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
        stop_accelerator(accelerator);
        vit_model_free(&model);
        vision_dataset_free(&train);
        vision_dataset_free(&validation);
        vision_dataset_free(&test);
        return 1;
    }
    printf("test accuracy=%.4f loss=%.6f seconds=%.2f samples/s=%.2f\n",
           metrics.accuracy, metrics.average_loss,
           metrics.elapsed_seconds,
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
    stop_accelerator(accelerator);
    return 0;
}

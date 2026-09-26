#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

typedef void (*OpsParallelRangeBody)(void *context, size_t begin, size_t end);

typedef struct {
    OpsParallelRangeBody body;
    void *context;
    size_t item_count;
    size_t items_per_task;
} OpsParallelTasks;

#ifdef __APPLE__
static void run_parallel_task(void *tasks_pointer, size_t task_index) {
    const OpsParallelTasks *tasks = tasks_pointer;
    const size_t begin = task_index * tasks->items_per_task;
    const size_t remaining = tasks->item_count - begin;
    const size_t end = begin + (remaining < tasks->items_per_task
                                    ? remaining
                                    : tasks->items_per_task);
    tasks->body(tasks->context, begin, end);
}
#endif

static void ops_parallel_for(size_t item_count, size_t minimum_items_per_task,
                             OpsParallelRangeBody body, void *context) {
    if (item_count == 0) {
        return;
    }
#ifdef __APPLE__
    const size_t maximum_task_count = 64;
    size_t items_per_task = (item_count + maximum_task_count - 1) /
                            maximum_task_count;
    if (items_per_task < minimum_items_per_task) {
        items_per_task = minimum_items_per_task;
    }
    const size_t task_count = (item_count + items_per_task - 1) / items_per_task;
    if (task_count > 1) {
        OpsParallelTasks tasks = {body, context, item_count, items_per_task};
        dispatch_apply_f(task_count,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                         &tasks, run_parallel_task);
        return;
    }
#else
    (void)minimum_items_per_task;
#endif
    body(context, 0, item_count);
}

static OpsDeviceBackend active_device_backend;
static int has_device_backend;

#define RUN_ON_DEVICE(operation, ...)                                        \
    (has_device_backend && active_device_backend.operation &&               \
     active_device_backend.operation(active_device_backend.context,          \
                                     __VA_ARGS__) == 0)

void ops_set_device_backend(const OpsDeviceBackend *backend) {
    ops_synchronize();
    if (backend) {
        active_device_backend = *backend;
        has_device_backend = 1;
    } else {
        has_device_backend = 0;
    }
}

void ops_synchronize(void) {
    if (has_device_backend && active_device_backend.synchronize) {
        active_device_backend.synchronize(active_device_backend.context);
    }
}

static OpsGemmBackend active_gemm_backend;
static OpsGemmBackwardBackend active_gemm_backward_backend;
static void *active_gemm_context;

static int is_matrix(const Tensor *tensor) {
    return tensor && tensor->ndim == 2 && tensor->data && tensor->grad;
}

void ops_set_gemm_backend(OpsGemmBackend backend, void *context) {
    active_gemm_backend = backend;
    active_gemm_context = context;
}

void ops_set_gemm_backward_backend(OpsGemmBackwardBackend backend,
                                   void *context) {
    active_gemm_backward_backend = backend;
    active_gemm_context = context;
}

void ops_reset_gemm_backend(void) {
    active_gemm_backend = NULL;
    active_gemm_backward_backend = NULL;
    active_gemm_context = NULL;
}

int ops_gemm(const Tensor *left, const Tensor *right, Tensor *output) {
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->cols != right->rows || output->rows != left->rows ||
        output->cols != right->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(gemm, left, right, output)) {
        return 0;
    }
    ops_synchronize();
    if (active_gemm_backend &&
        active_gemm_backend(active_gemm_context, left->data, right->data,
                            output->data, left->rows, left->cols,
                            right->cols) == 0) {
        return 0;
    }
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t col = 0; col < output->cols; ++col) {
            float value = 0.0f;
            for (size_t i = 0; i < left->cols; ++i) {
                value += left->data[row * left->cols + i] *
                         right->data[i * right->cols + col];
            }
            output->data[row * output->cols + col] = value;
        }
    }
    return 0;
}

int ops_gemm_backward(Tensor *left, Tensor *right, const Tensor *output) {
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->cols != right->rows || output->rows != left->rows ||
        output->cols != right->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(gemm_backward, left, right, output)) {
        return 0;
    }
    ops_synchronize();
    if (active_gemm_backward_backend &&
        active_gemm_backward_backend(
            active_gemm_context, left->data, right->data, output->grad,
            left->grad, right->grad, left->rows, left->cols,
            right->cols) == 0) {
        return 0;
    }
    for (size_t row = 0; row < left->rows; ++row) {
        for (size_t i = 0; i < left->cols; ++i) {
            for (size_t col = 0; col < right->cols; ++col) {
                const float gradient = output->grad[row * output->cols + col];
                left->grad[row * left->cols + i] +=
                    gradient * right->data[i * right->cols + col];
                right->grad[i * right->cols + col] +=
                    left->data[row * left->cols + i] * gradient;
            }
        }
    }
    return 0;
}

int ops_gemm_transposed_left(const Tensor *left, const Tensor *right,
                             Tensor *output) {
    ops_synchronize();
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->rows != right->rows || output->rows != left->cols ||
        output->cols != right->cols) {
        return -1;
    }
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t col = 0; col < output->cols; ++col) {
            float value = 0.0f;
            for (size_t i = 0; i < left->rows; ++i) {
                value += left->data[i * left->cols + row] *
                         right->data[i * right->cols + col];
            }
            output->data[row * output->cols + col] = value;
        }
    }
    return 0;
}

int ops_gemm_transposed_left_backward(Tensor *left, Tensor *right,
                                      const Tensor *output) {
    ops_synchronize();
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->rows != right->rows || output->rows != left->cols ||
        output->cols != right->cols) {
        return -1;
    }
    for (size_t i = 0; i < left->rows; ++i) {
        for (size_t row = 0; row < left->cols; ++row) {
            for (size_t col = 0; col < right->cols; ++col) {
                const float gradient = output->grad[row * output->cols + col];
                left->grad[i * left->cols + row] +=
                    right->data[i * right->cols + col] * gradient;
                right->grad[i * right->cols + col] +=
                    left->data[i * left->cols + row] * gradient;
            }
        }
    }
    return 0;
}

int ops_residual(const Tensor *left, const Tensor *right, Tensor *output) {
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->rows != right->rows || left->cols != right->cols ||
        output->rows != left->rows || output->cols != left->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(residual, left, right, output)) {
        return 0;
    }
    ops_synchronize();
    for (size_t i = 0; i < left->rows * left->cols; ++i) {
        output->data[i] = left->data[i] + right->data[i];
    }
    return 0;
}

int ops_residual_backward(Tensor *left, Tensor *right, const Tensor *output) {
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->rows != right->rows || left->cols != right->cols ||
        output->rows != left->rows || output->cols != left->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(residual_backward, left, right, output)) {
        return 0;
    }
    ops_synchronize();

    for (size_t i = 0; i < left->rows * left->cols; ++i) {
        left->grad[i] += output->grad[i];
        right->grad[i] += output->grad[i];
    }
    return 0;
}

typedef struct {
    const Tensor *input;
    const Tensor *bias;
    Tensor *output;
} BiasAddContext;

static void bias_add_rows(void *context_pointer, size_t begin, size_t end) {
    const BiasAddContext *context = context_pointer;
    const size_t cols = context->input->cols;
    for (size_t row = begin; row < end; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            context->output->data[row * cols + col] =
                context->input->data[row * cols + col] +
                context->bias->data[col];
        }
    }
}

int ops_bias_add(const Tensor *input, const Tensor *bias, Tensor *output) {
    if (!is_matrix(input) || !is_matrix(bias) || !is_matrix(output) ||
        bias->rows != 1 || bias->cols != input->cols ||
        output->rows != input->rows || output->cols != input->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(bias_add, input, bias, output)) {
        return 0;
    }
    ops_synchronize();
    BiasAddContext context = {input, bias, output};
    ops_parallel_for(input->rows, 64, bias_add_rows, &context);
    return 0;
}

static int valid_head_layout(const Tensor *combined, const Tensor *query,
                             const Tensor *key, const Tensor *value,
                             size_t combined_width, size_t batch,
                             size_t sequence_length, size_t d_model,
                             size_t heads) {
    if (heads == 0 || d_model % heads != 0) {
        return 0;
    }
    const size_t rows = batch * sequence_length;
    const size_t head_dimension = d_model / heads;
    int valid = is_matrix(combined) && combined->rows == rows &&
                combined->cols == combined_width;
    const Tensor *per_head[3] = {query, key, value};
    for (size_t i = 0; i < 3 && valid; ++i) {
        valid = !per_head[i] ||
                (is_matrix(per_head[i]) && per_head[i]->rows == rows * heads &&
                 per_head[i]->cols == head_dimension);
    }
    return valid;
}

static size_t head_row(size_t sample, size_t head, size_t position,
                       size_t heads, size_t sequence_length) {
    return (sample * heads + head) * sequence_length + position;
}

int ops_accumulate_gradient(Tensor *destination, const Tensor *source) {
    if (!is_matrix(destination) || !is_matrix(source) ||
        destination->rows != source->rows || destination->cols != source->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(accumulate_gradient, destination, source)) {
        return 0;
    }
    ops_synchronize();
    for (size_t i = 0; i < destination->rows * destination->cols; ++i) {
        destination->grad[i] += source->grad[i];
    }
    return 0;
}

int ops_zero_gradient(Tensor *tensor) {
    if (!is_matrix(tensor)) {
        return -1;
    }
    if (RUN_ON_DEVICE(zero_gradient, tensor)) {
        return 0;
    }
    ops_synchronize();
    memset(tensor->grad, 0, tensor_numel(tensor) * sizeof(float));
    return 0;
}

int ops_zero_gradients(Tensor *const *tensors, size_t tensor_count) {
    if (!tensors) {
        return -1;
    }
    for (size_t i = 0; i < tensor_count; ++i) {
        if (!is_matrix(tensors[i])) {
            return -1;
        }
    }
    if (tensor_count == 0 ||
        RUN_ON_DEVICE(zero_gradients, tensors, tensor_count)) {
        return 0;
    }
    ops_synchronize();
    for (size_t i = 0; i < tensor_count; ++i) {
        memset(tensors[i]->grad, 0, tensor_numel(tensors[i]) * sizeof(float));
    }
    return 0;
}

int ops_upload_values(Tensor *tensor, const float *values) {
    if (!is_matrix(tensor) || !values) {
        return -1;
    }
    ops_synchronize();
    memcpy(tensor->data, values, tensor_numel(tensor) * sizeof(float));
    return 0;
}

int ops_upload_labels(Tensor *tensor, const size_t *labels) {
    if (!is_matrix(tensor) || tensor->cols != 1 || !labels) {
        return -1;
    }
    ops_synchronize();
    for (size_t row = 0; row < tensor->rows; ++row) {
        tensor->data[row] = (float)labels[row];
    }
    return 0;
}

static int valid_patch_layout(const Tensor *patches, size_t batch,
                              size_t channels, size_t height, size_t width,
                              size_t patch_size) {
    return patch_size > 0 && height % patch_size == 0 &&
           width % patch_size == 0 && is_matrix(patches) &&
           patches->rows == batch * (height / patch_size) * (width / patch_size) &&
           patches->cols == channels * patch_size * patch_size;
}

static size_t image_index_of_patch_element(size_t patch_row, size_t element,
                                           size_t channels, size_t height,
                                           size_t width, size_t patch_size) {
    const size_t patches_per_row = width / patch_size;
    const size_t patches_per_image = (height / patch_size) * patches_per_row;
    const size_t sample = patch_row / patches_per_image;
    const size_t patch_index = patch_row % patches_per_image;
    const size_t channel = element / (patch_size * patch_size);
    const size_t y = (patch_index / patches_per_row) * patch_size +
                     (element / patch_size) % patch_size;
    const size_t x = (patch_index % patches_per_row) * patch_size +
                     element % patch_size;
    return ((sample * channels + channel) * height + y) * width + x;
}

int ops_extract_patches(const Tensor *images, Tensor *patches, size_t channels,
                        size_t height, size_t width, size_t patch_size) {
    if (!is_matrix(images) ||
        images->cols != channels * height * width ||
        !valid_patch_layout(patches, images->rows, channels, height, width,
                            patch_size)) {
        return -1;
    }
    if (RUN_ON_DEVICE(extract_patches, images, patches, channels, height,
                      width, patch_size)) {
        return 0;
    }
    ops_synchronize();
    for (size_t row = 0; row < patches->rows; ++row) {
        for (size_t element = 0; element < patches->cols; ++element) {
            patches->data[row * patches->cols + element] =
                images->data[image_index_of_patch_element(
                    row, element, channels, height, width, patch_size)];
        }
    }
    return 0;
}

int ops_extract_patches_backward(const Tensor *patches, float *image_grad,
                                 size_t channels, size_t height, size_t width,
                                 size_t patch_size) {
    if (!image_grad || patches->cols != channels * patch_size * patch_size) {
        return -1;
    }
    ops_synchronize();
    for (size_t row = 0; row < patches->rows; ++row) {
        for (size_t element = 0; element < patches->cols; ++element) {
            image_grad[image_index_of_patch_element(
                row, element, channels, height, width, patch_size)] +=
                patches->grad[row * patches->cols + element];
        }
    }
    return 0;
}

static int valid_token_layout(const Tensor *patch_tokens,
                              const Tensor *cls_token,
                              const Tensor *positional, const Tensor *tokens,
                              size_t batch) {
    if (!is_matrix(patch_tokens) || !is_matrix(cls_token) ||
        !is_matrix(positional) || !is_matrix(tokens) || batch == 0 ||
        patch_tokens->rows % batch != 0) {
        return 0;
    }
    const size_t sequence_length = patch_tokens->rows / batch + 1;
    const size_t d_model = patch_tokens->cols;
    return cls_token->rows == 1 && cls_token->cols == d_model &&
           positional->rows == sequence_length && positional->cols == d_model &&
           tokens->rows == batch * sequence_length && tokens->cols == d_model;
}

int ops_token_embedding(const Tensor *patch_tokens, const Tensor *cls_token,
                        const Tensor *positional, Tensor *tokens,
                        size_t batch) {
    if (!valid_token_layout(patch_tokens, cls_token, positional, tokens,
                            batch)) {
        return -1;
    }
    if (RUN_ON_DEVICE(token_embedding, patch_tokens, cls_token, positional,
                      tokens, batch)) {
        return 0;
    }
    ops_synchronize();
    const size_t d_model = tokens->cols;
    const size_t sequence_length = positional->rows;
    const size_t patch_count = sequence_length - 1;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const float *source = position == 0
                ? cls_token->data
                : patch_tokens->data +
                  (sample * patch_count + position - 1) * d_model;
            float *destination =
                tokens->data + (sample * sequence_length + position) * d_model;
            for (size_t dimension = 0; dimension < d_model; ++dimension) {
                destination[dimension] = source[dimension] +
                    positional->data[position * d_model + dimension];
            }
        }
    }
    return 0;
}

int ops_token_embedding_backward(Tensor *patch_tokens, Tensor *cls_token,
                                 Tensor *positional, const Tensor *tokens,
                                 size_t batch) {
    if (!valid_token_layout(patch_tokens, cls_token, positional, tokens,
                            batch)) {
        return -1;
    }
    if (RUN_ON_DEVICE(token_embedding_backward, patch_tokens, cls_token,
                      positional, tokens, batch)) {
        return 0;
    }
    ops_synchronize();
    const size_t d_model = tokens->cols;
    const size_t sequence_length = positional->rows;
    const size_t patch_count = sequence_length - 1;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const float *upstream =
                tokens->grad + (sample * sequence_length + position) * d_model;
            float *destination = position == 0
                ? cls_token->grad
                : patch_tokens->grad +
                  (sample * patch_count + position - 1) * d_model;
            for (size_t dimension = 0; dimension < d_model; ++dimension) {
                positional->grad[position * d_model + dimension] +=
                    upstream[dimension];
                destination[dimension] += upstream[dimension];
            }
        }
    }
    return 0;
}

int ops_gather_rows(const Tensor *source, Tensor *destination,
                    size_t row_stride) {
    if (!is_matrix(source) || !is_matrix(destination) || row_stride == 0 ||
        source->cols != destination->cols ||
        source->rows != destination->rows * row_stride) {
        return -1;
    }
    if (RUN_ON_DEVICE(gather_rows, source, destination, row_stride)) {
        return 0;
    }
    ops_synchronize();
    for (size_t row = 0; row < destination->rows; ++row) {
        memcpy(destination->data + row * destination->cols,
               source->data + row * row_stride * source->cols,
               destination->cols * sizeof(float));
    }
    return 0;
}

int ops_gather_rows_backward(Tensor *source, const Tensor *destination,
                             size_t row_stride) {
    if (!is_matrix(source) || !is_matrix(destination) || row_stride == 0 ||
        source->cols != destination->cols ||
        source->rows != destination->rows * row_stride) {
        return -1;
    }
    if (RUN_ON_DEVICE(gather_rows_backward, source, destination, row_stride)) {
        return 0;
    }
    ops_synchronize();
    for (size_t row = 0; row < destination->rows; ++row) {
        for (size_t col = 0; col < destination->cols; ++col) {
            source->grad[row * row_stride * source->cols + col] +=
                destination->grad[row * destination->cols + col];
        }
    }
    return 0;
}

int ops_softmax_cross_entropy_tensor(const Tensor *logits,
                                     const Tensor *targets,
                                     const Tensor *class_weights,
                                     float label_smoothing, Tensor *loss,
                                     Tensor *logits_grad) {
    if (!is_matrix(logits) || !is_matrix(targets) || !is_matrix(loss) ||
        !is_matrix(logits_grad) || targets->rows != logits->rows ||
        targets->cols != 1 || loss->rows != 1 || loss->cols != 1 ||
        logits_grad->rows != logits->rows || logits_grad->cols != logits->cols ||
        (class_weights && (!is_matrix(class_weights) ||
                           class_weights->rows != 1 ||
                           class_weights->cols != logits->cols)) ||
        label_smoothing < 0.0f || label_smoothing >= 1.0f) {
        return -1;
    }
    if (RUN_ON_DEVICE(softmax_cross_entropy, logits, targets, class_weights,
                      label_smoothing, loss, logits_grad)) {
        return 0;
    }
    ops_synchronize();
    size_t *labels = malloc(targets->rows * sizeof(*labels));
    if (!labels) {
        return -1;
    }
    for (size_t row = 0; row < targets->rows; ++row) {
        labels[row] = (size_t)targets->data[row];
    }
    const int result = ops_softmax_cross_entropy_smoothed(
        logits, labels, class_weights ? class_weights->data : NULL,
        label_smoothing, &loss->data[0], logits_grad);
    free(labels);
    return result;
}

int ops_clip_gradient_norm(Parameter *const *parameters,
                           size_t parameter_count, float max_norm) {
    if (!parameters || parameter_count == 0 || max_norm <= 0.0f) {
        return -1;
    }
    if (RUN_ON_DEVICE(clip_gradient_norm, parameters, parameter_count,
                      max_norm)) {
        return 0;
    }
    ops_synchronize();
    double squared_norm = 0.0;
    for (size_t p = 0; p < parameter_count; ++p) {
        const Tensor *value = &parameters[p]->value;
        for (size_t i = 0; i < tensor_numel(value); ++i) {
            squared_norm += (double)value->grad[i] * value->grad[i];
        }
    }
    const double norm = sqrt(squared_norm);
    if (norm <= (double)max_norm || norm == 0.0) {
        return 0;
    }
    const float scale = (float)((double)max_norm / norm);
    for (size_t p = 0; p < parameter_count; ++p) {
        Tensor *value = &parameters[p]->value;
        for (size_t i = 0; i < tensor_numel(value); ++i) {
            value->grad[i] *= scale;
        }
    }
    return 0;
}

int ops_split_heads(const Tensor *projected, Tensor *query, Tensor *key,
                    Tensor *value, size_t batch, size_t sequence_length,
                    size_t d_model, size_t heads) {
    if (!query || !key || !value ||
        !valid_head_layout(projected, query, key, value, 3 * d_model, batch,
                           sequence_length, d_model, heads)) {
        return -1;
    }
    if (RUN_ON_DEVICE(split_heads, projected, query, key, value, batch,
                      sequence_length, d_model, heads)) {
        return 0;
    }
    ops_synchronize();
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const float *source = projected->data +
                (sample * sequence_length + position) * 3 * d_model;
            for (size_t head = 0; head < heads; ++head) {
                const size_t target =
                    head_row(sample, head, position, heads, sequence_length) *
                    head_dimension;
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    const size_t offset = head * head_dimension + dimension;
                    query->data[target + dimension] = source[offset];
                    key->data[target + dimension] = source[d_model + offset];
                    value->data[target + dimension] =
                        source[2 * d_model + offset];
                }
            }
        }
    }
    return 0;
}

int ops_split_heads_backward(Tensor *projected, const Tensor *query,
                             const Tensor *key, const Tensor *value,
                             size_t batch, size_t sequence_length,
                             size_t d_model, size_t heads) {
    if (!query || !key || !value ||
        !valid_head_layout(projected, query, key, value, 3 * d_model, batch,
                           sequence_length, d_model, heads)) {
        return -1;
    }
    if (RUN_ON_DEVICE(split_heads_backward, projected, query, key, value,
                      batch, sequence_length, d_model, heads)) {
        return 0;
    }
    ops_synchronize();
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            float *destination = projected->grad +
                (sample * sequence_length + position) * 3 * d_model;
            for (size_t head = 0; head < heads; ++head) {
                const size_t source =
                    head_row(sample, head, position, heads, sequence_length) *
                    head_dimension;
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    const size_t offset = head * head_dimension + dimension;
                    destination[offset] += query->grad[source + dimension];
                    destination[d_model + offset] +=
                        key->grad[source + dimension];
                    destination[2 * d_model + offset] +=
                        value->grad[source + dimension];
                }
            }
        }
    }
    return 0;
}

int ops_merge_heads(const Tensor *attended, Tensor *merged, size_t batch,
                    size_t sequence_length, size_t d_model, size_t heads) {
    if (!valid_head_layout(merged, attended, NULL, NULL, d_model, batch,
                           sequence_length, d_model, heads)) {
        return -1;
    }
    if (RUN_ON_DEVICE(merge_heads, attended, merged, batch, sequence_length,
                      d_model, heads)) {
        return 0;
    }
    ops_synchronize();
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            float *destination = merged->data +
                (sample * sequence_length + position) * d_model;
            for (size_t head = 0; head < heads; ++head) {
                const float *source = attended->data +
                    head_row(sample, head, position, heads, sequence_length) *
                    head_dimension;
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    destination[head * head_dimension + dimension] =
                        source[dimension];
                }
            }
        }
    }
    return 0;
}

int ops_merge_heads_backward(Tensor *attended, const Tensor *merged,
                             size_t batch, size_t sequence_length,
                             size_t d_model, size_t heads) {
    if (!valid_head_layout(merged, attended, NULL, NULL, d_model, batch,
                           sequence_length, d_model, heads)) {
        return -1;
    }
    if (RUN_ON_DEVICE(merge_heads_backward, attended, merged, batch,
                      sequence_length, d_model, heads)) {
        return 0;
    }
    ops_synchronize();
    const size_t head_dimension = d_model / heads;
    for (size_t sample = 0; sample < batch; ++sample) {
        for (size_t position = 0; position < sequence_length; ++position) {
            const float *source = merged->grad +
                (sample * sequence_length + position) * d_model;
            for (size_t head = 0; head < heads; ++head) {
                float *destination = attended->grad +
                    head_row(sample, head, position, heads, sequence_length) *
                    head_dimension;
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    destination[dimension] +=
                        source[head * head_dimension + dimension];
                }
            }
        }
    }
    return 0;
}

int ops_bias_add_backward(const Tensor *output, Tensor *bias_grad) {
    if (!is_matrix(output) || !is_matrix(bias_grad) ||
        bias_grad->rows != 1 || bias_grad->cols != output->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(bias_add_backward, output, bias_grad)) {
        return 0;
    }
    ops_synchronize();
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t col = 0; col < output->cols; ++col) {
            bias_grad->grad[col] += output->grad[row * output->cols + col];
        }
    }
    return 0;
}

static void row_mean_and_inverse_std(const float *row_values, size_t cols,
                                     float epsilon, float *mean,
                                     float *inverse_std) {
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        sum += row_values[col];
    }
    const float row_mean = sum / (float)cols;
    float variance = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const float centered = row_values[col] - row_mean;
        variance += centered * centered;
    }
    variance /= (float)cols;
    *mean = row_mean;
    *inverse_std = 1.0f / sqrtf(variance + epsilon);
}

typedef struct {
    const Tensor *input;
    const Tensor *gamma;
    const Tensor *beta;
    float epsilon;
    Tensor *output;
} LayerNormContext;

static void layer_norm_rows(void *context_pointer, size_t begin, size_t end) {
    const LayerNormContext *context = context_pointer;
    const size_t cols = context->input->cols;
    for (size_t row = begin; row < end; ++row) {
        const float *row_values = context->input->data + row * cols;
        float mean = 0.0f;
        float inverse_std = 0.0f;
        row_mean_and_inverse_std(row_values, cols, context->epsilon, &mean,
                                 &inverse_std);
        for (size_t col = 0; col < cols; ++col) {
            const float normalized = (row_values[col] - mean) * inverse_std;
            context->output->data[row * cols + col] =
                normalized * context->gamma->data[col] +
                context->beta->data[col];
        }
    }
}

int ops_layer_norm(const Tensor *input, const Tensor *gamma, const Tensor *beta,
                   float epsilon, Tensor *output) {
    if (!is_matrix(input) || !is_matrix(gamma) || !is_matrix(beta) ||
        !is_matrix(output) || gamma->rows != 1 || beta->rows != 1 ||
        gamma->cols != input->cols || beta->cols != input->cols ||
        output->rows != input->rows || output->cols != input->cols ||
        epsilon <= 0.0f) {
        return -1;
    }
    if (RUN_ON_DEVICE(layer_norm, input, gamma, beta, epsilon, output)) {
        return 0;
    }
    ops_synchronize();
    LayerNormContext context = {input, gamma, beta, epsilon, output};
    ops_parallel_for(input->rows, 32, layer_norm_rows, &context);
    return 0;
}

typedef struct {
    const Tensor *input;
    const Tensor *gamma;
    float epsilon;
    const Tensor *output;
    Tensor *input_grad;
    float *row_means;
    float *row_inverse_stds;
} LayerNormBackwardContext;

static void layer_norm_backward_rows(void *context_pointer, size_t begin,
                                     size_t end) {
    const LayerNormBackwardContext *context = context_pointer;
    const size_t cols = context->input->cols;
    for (size_t row = begin; row < end; ++row) {
        const float *row_values = context->input->data + row * cols;
        const float *upstream = context->output->grad + row * cols;
        float mean = 0.0f;
        float inverse_std = 0.0f;
        row_mean_and_inverse_std(row_values, cols, context->epsilon, &mean,
                                 &inverse_std);
        context->row_means[row] = mean;
        context->row_inverse_stds[row] = inverse_std;
        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float normalized = (row_values[col] - mean) * inverse_std;
            const float dxhat = upstream[col] * context->gamma->data[col];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * normalized;
        }
        for (size_t col = 0; col < cols; ++col) {
            const float normalized = (row_values[col] - mean) * inverse_std;
            const float dxhat = upstream[col] * context->gamma->data[col];
            context->input_grad->grad[row * cols + col] +=
                inverse_std * (dxhat - sum_dxhat / (float)cols -
                               normalized * sum_dxhat_xhat / (float)cols);
        }
    }
}

typedef struct {
    const Tensor *input;
    const Tensor *output;
    const float *row_means;
    const float *row_inverse_stds;
    Tensor *gamma_grad;
    Tensor *beta_grad;
} LayerNormParameterGradientContext;

static void layer_norm_parameter_gradient_columns(void *context_pointer,
                                                  size_t begin, size_t end) {
    const LayerNormParameterGradientContext *context = context_pointer;
    const size_t rows = context->input->rows;
    const size_t cols = context->input->cols;
    for (size_t col = begin; col < end; ++col) {
        float gamma_gradient = 0.0f;
        float beta_gradient = 0.0f;
        for (size_t row = 0; row < rows; ++row) {
            const float upstream = context->output->grad[row * cols + col];
            const float normalized =
                (context->input->data[row * cols + col] -
                 context->row_means[row]) * context->row_inverse_stds[row];
            gamma_gradient += upstream * normalized;
            beta_gradient += upstream;
        }
        context->gamma_grad->grad[col] += gamma_gradient;
        context->beta_grad->grad[col] += beta_gradient;
    }
}

int ops_layer_norm_backward(const Tensor *input, const Tensor *gamma,
                            float epsilon, const Tensor *output,
                            Tensor *input_grad, Tensor *gamma_grad,
                            Tensor *beta_grad) {
    if (!is_matrix(input) || !is_matrix(gamma) || !is_matrix(output) ||
        !is_matrix(input_grad) || !is_matrix(gamma_grad) ||
        !is_matrix(beta_grad) || gamma->rows != 1 || gamma->cols != input->cols ||
        output->rows != input->rows || output->cols != input->cols ||
        input_grad->rows != input->rows || input_grad->cols != input->cols ||
        gamma_grad->rows != 1 || gamma_grad->cols != input->cols ||
        beta_grad->rows != 1 || beta_grad->cols != input->cols ||
        epsilon <= 0.0f) {
        return -1;
    }
    if (RUN_ON_DEVICE(layer_norm_backward, input, gamma, epsilon, output,
                      input_grad, gamma_grad, beta_grad)) {
        return 0;
    }
    ops_synchronize();
    float *row_statistics = malloc(2 * input->rows * sizeof(*row_statistics));
    if (!row_statistics) {
        return -1;
    }
    float *row_means = row_statistics;
    float *row_inverse_stds = row_statistics + input->rows;
    LayerNormBackwardContext row_context = {
        input, gamma, epsilon, output, input_grad, row_means, row_inverse_stds,
    };
    ops_parallel_for(input->rows, 32, layer_norm_backward_rows, &row_context);
    LayerNormParameterGradientContext parameter_context = {
        input, output, row_means, row_inverse_stds, gamma_grad, beta_grad,
    };
    ops_parallel_for(input->cols, 8, layer_norm_parameter_gradient_columns,
                     &parameter_context);
    free(row_statistics);
    return 0;
}

static const float gelu_cubic_coefficient = 0.044715f;

static float gelu_tanh_scale(void) {
    return sqrtf(2.0f / 3.14159265358979323846f);
}

typedef struct {
    const Tensor *input;
    const Tensor *output;
    float tanh_scale;
} GeluContext;

static void gelu_elements(void *context_pointer, size_t begin, size_t end) {
    const GeluContext *context = context_pointer;
    for (size_t i = begin; i < end; ++i) {
        const float x = context->input->data[i];
        context->output->data[i] = 0.5f * x *
            (1.0f + tanhf(context->tanh_scale *
                          (x + gelu_cubic_coefficient * x * x * x)));
    }
}

int ops_gelu(const Tensor *input, Tensor *output) {
    if (!is_matrix(input) || !is_matrix(output) ||
        input->rows != output->rows || input->cols != output->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(gelu, input, output)) {
        return 0;
    }
    ops_synchronize();
    GeluContext context = {input, output, gelu_tanh_scale()};
    ops_parallel_for(input->rows * input->cols, 4096, gelu_elements, &context);
    return 0;
}

static void gelu_backward_elements(void *context_pointer, size_t begin,
                                   size_t end) {
    const GeluContext *context = context_pointer;
    const float tanh_scale = context->tanh_scale;
    for (size_t i = begin; i < end; ++i) {
        const float x = context->input->data[i];
        const float inner = tanh_scale * (x + gelu_cubic_coefficient * x * x * x);
        const float tanh_inner = tanhf(inner);
        const float derivative = 0.5f * (1.0f + tanh_inner) +
                                 0.5f * x * (1.0f - tanh_inner * tanh_inner) *
                                 tanh_scale *
                                 (1.0f + 3.0f * gelu_cubic_coefficient * x * x);
        context->input->grad[i] += context->output->grad[i] * derivative;
    }
}

int ops_gelu_backward(const Tensor *input, const Tensor *output) {
    if (!is_matrix(input) || !is_matrix(output) ||
        input->rows != output->rows || input->cols != output->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(gelu_backward, input, output)) {
        return 0;
    }
    ops_synchronize();
    GeluContext context = {input, output, gelu_tanh_scale()};
    ops_parallel_for(input->rows * input->cols, 4096, gelu_backward_elements,
                     &context);
    return 0;
}

int ops_softmax_cross_entropy(const Tensor *logits, const size_t *targets,
                              float *loss, Tensor *logits_grad) {
    return ops_softmax_cross_entropy_weighted(logits, targets, NULL, loss,
                                              logits_grad);
}

int ops_softmax_cross_entropy_weighted(const Tensor *logits,
                                       const size_t *targets,
                                       const float *class_weights,
                                       float *loss, Tensor *logits_grad) {
    return ops_softmax_cross_entropy_smoothed(logits, targets, class_weights,
                                              0.0f, loss, logits_grad);
}

int ops_softmax_cross_entropy_smoothed(const Tensor *logits,
                                       const size_t *targets,
                                       const float *class_weights,
                                       float label_smoothing, float *loss,
                                       Tensor *logits_grad) {
    ops_synchronize();
    if (!is_matrix(logits) || !targets || !loss || !is_matrix(logits_grad) ||
        logits_grad->rows != logits->rows || logits_grad->cols != logits->cols ||
        label_smoothing < 0.0f || label_smoothing >= 1.0f) {
        return -1;
    }
    const float off_target_probability = label_smoothing / (float)logits->cols;
    const float on_target_probability =
        1.0f - label_smoothing + off_target_probability;
    *loss = 0.0f;
    for (size_t row = 0; row < logits->rows; ++row) {
        if (targets[row] >= logits->cols) {
            return -1;
        }
        float maximum = -INFINITY;
        for (size_t col = 0; col < logits->cols; ++col) {
            if (logits->data[row * logits->cols + col] > maximum) {
                maximum = logits->data[row * logits->cols + col];
            }
        }
        float denominator = 0.0f;
        for (size_t col = 0; col < logits->cols; ++col) {
            denominator += expf(logits->data[row * logits->cols + col] - maximum);
        }
        const float log_denominator = maximum + logf(denominator);
        const float weight = class_weights ? class_weights[targets[row]] : 1.0f;
        for (size_t col = 0; col < logits->cols; ++col) {
            const float logit = logits->data[row * logits->cols + col];
            const float target_probability = col == targets[row]
                ? on_target_probability
                : off_target_probability;
            const float probability = expf(logit - log_denominator);
            *loss += weight * target_probability * (log_denominator - logit);
            logits_grad->grad[row * logits->cols + col] +=
                weight * (probability - target_probability);
        }
    }
    *loss /= (float)logits->rows;
    for (size_t i = 0; i < logits->rows * logits->cols; ++i) {
        logits_grad->grad[i] /= (float)logits->rows;
    }
    return 0;
}

static int valid_adamw_hyperparameters(float learning_rate, float beta1,
                                       float beta2, float epsilon) {
    return learning_rate > 0.0f && beta1 >= 0.0f && beta1 < 1.0f &&
           beta2 >= 0.0f && beta2 < 1.0f && epsilon > 0.0f;
}

static float adamw_bias_correction(float beta, size_t step) {
    return 1.0f - powf(beta, (float)step);
}

static void apply_adamw_update(Parameter *parameter, float learning_rate,
                               float beta1, float beta2, float epsilon,
                               float weight_decay) {
    const float correction1 = adamw_bias_correction(beta1, parameter->step);
    const float correction2 = adamw_bias_correction(beta2, parameter->step);
    for (size_t i = 0; i < tensor_numel(&parameter->value); ++i) {
        const float gradient = parameter->value.grad[i];
        parameter->m[i] = beta1 * parameter->m[i] + (1.0f - beta1) * gradient;
        parameter->v[i] = beta2 * parameter->v[i] +
                         (1.0f - beta2) * gradient * gradient;
        const float estimate = (parameter->m[i] / correction1) /
                               (sqrtf(parameter->v[i] / correction2) + epsilon);
        parameter->value.data[i] -=
            learning_rate * (estimate + weight_decay * parameter->value.data[i]);
    }
    parameter_zero_grad(parameter);
}

int ops_adamw_step(Parameter *parameter, float learning_rate,
                   float beta1, float beta2, float epsilon,
                   float weight_decay) {
    if (!parameter || !parameter->value.data || !parameter->value.grad ||
        !parameter->m || !parameter->v || weight_decay < 0.0f ||
        !valid_adamw_hyperparameters(learning_rate, beta1, beta2, epsilon)) {
        return -1;
    }
    parameter->step += 1;
    if (RUN_ON_DEVICE(adamw_step, parameter, learning_rate, beta1, beta2,
                      epsilon, weight_decay,
                      adamw_bias_correction(beta1, parameter->step),
                      adamw_bias_correction(beta2, parameter->step))) {
        return 0;
    }
    ops_synchronize();
    apply_adamw_update(parameter, learning_rate, beta1, beta2, epsilon,
                       weight_decay);
    return 0;
}

int ops_adamw_step_parameters(Parameter *const *parameters,
                              const float *weight_decays,
                              size_t parameter_count, float learning_rate,
                              float beta1, float beta2, float epsilon,
                              float max_gradient_norm) {
    if (!parameters || !weight_decays || parameter_count == 0 ||
        !valid_adamw_hyperparameters(learning_rate, beta1, beta2, epsilon) ||
        max_gradient_norm < 0.0f) {
        return -1;
    }
    for (size_t i = 0; i < parameter_count; ++i) {
        const Parameter *parameter = parameters[i];
        if (!parameter || !parameter->value.data || !parameter->value.grad ||
            !parameter->m || !parameter->v || weight_decays[i] < 0.0f) {
            return -1;
        }
    }
    for (size_t i = 0; i < parameter_count; ++i) {
        parameters[i]->step += 1;
    }
    if (RUN_ON_DEVICE(adamw_step_parameters, parameters, weight_decays,
                      parameter_count, learning_rate, beta1, beta2, epsilon,
                      max_gradient_norm)) {
        return 0;
    }
    ops_synchronize();
    if (max_gradient_norm > 0.0f &&
        ops_clip_gradient_norm(parameters, parameter_count,
                               max_gradient_norm) != 0) {
        return -1;
    }
    for (size_t p = 0; p < parameter_count; ++p) {
        apply_adamw_update(parameters[p], learning_rate, beta1, beta2, epsilon,
                           weight_decays[p]);
    }
    return 0;
}

int ops_causal_attention(const Tensor *query, const Tensor *key,
                         const Tensor *value, float scale,
                         Tensor *probabilities, Tensor *output) {
    ops_synchronize();
    if (!is_matrix(query) || !is_matrix(key) || !is_matrix(value) ||
        !is_matrix(probabilities) || !is_matrix(output) || scale <= 0.0f ||
        query->rows != key->rows || key->rows != value->rows ||
        query->cols != key->cols || probabilities->rows != query->rows ||
        probabilities->cols != key->rows || output->rows != query->rows ||
        output->cols != value->cols) {
        return -1;
    }
    const size_t length = query->rows;
    for (size_t row = 0; row < length; ++row) {
        float maximum = -INFINITY;
        for (size_t col = 0; col < length; ++col) {
            float score = -INFINITY;
            if (col <= row) {
                score = 0.0f;
                for (size_t dimension = 0; dimension < query->cols; ++dimension) {
                    score += query->data[row * query->cols + dimension] *
                             key->data[col * key->cols + dimension];
                }
                score *= scale;
                if (score > maximum) {
                    maximum = score;
                }
            }
            probabilities->data[row * probabilities->cols + col] = score;
        }
        float denominator = 0.0f;
        for (size_t col = 0; col < length; ++col) {
            if (col <= row) {
                probabilities->data[row * probabilities->cols + col] =
                    expf(probabilities->data[row * probabilities->cols + col] -
                         maximum);
                denominator += probabilities->data[row * probabilities->cols + col];
            } else {
                probabilities->data[row * probabilities->cols + col] = 0.0f;
            }
        }
        for (size_t col = 0; col < length; ++col) {
            probabilities->data[row * probabilities->cols + col] /= denominator;
        }
        for (size_t dimension = 0; dimension < output->cols; ++dimension) {
            float result = 0.0f;
            for (size_t col = 0; col <= row; ++col) {
                result += probabilities->data[row * probabilities->cols + col] *
                          value->data[col * value->cols + dimension];
            }
            output->data[row * output->cols + dimension] = result;
        }
    }
    return 0;
}

int ops_causal_attention_backward(const Tensor *query, const Tensor *key,
                                  const Tensor *value, float scale,
                                  const Tensor *probabilities,
                                  const Tensor *output, Tensor *query_grad,
                                  Tensor *key_grad, Tensor *value_grad) {
    ops_synchronize();
    if (!is_matrix(query) || !is_matrix(key) || !is_matrix(value) ||
        !is_matrix(probabilities) || !is_matrix(output) ||
        !is_matrix(query_grad) || !is_matrix(key_grad) ||
        !is_matrix(value_grad) || scale <= 0.0f ||
        query->rows != key->rows || key->rows != value->rows ||
        query->cols != key->cols || probabilities->rows != query->rows ||
        probabilities->cols != key->rows || output->rows != query->rows ||
        output->cols != value->cols || query_grad->rows != query->rows ||
        query_grad->cols != query->cols || key_grad->rows != key->rows ||
        key_grad->cols != key->cols || value_grad->rows != value->rows ||
        value_grad->cols != value->cols) {
        return -1;
    }
    const size_t length = query->rows;
    for (size_t row = 0; row < length; ++row) {
        for (size_t col = 0; col <= row; ++col) {
            float probability_gradient = 0.0f;
            for (size_t dimension = 0; dimension < output->cols; ++dimension) {
                probability_gradient +=
                    output->grad[row * output->cols + dimension] *
                    value->data[col * value->cols + dimension];
                value_grad->grad[col * value->cols + dimension] +=
                    probabilities->data[row * probabilities->cols + col] *
                    output->grad[row * output->cols + dimension];
            }
            float softmax_gradient = probability_gradient *
                probabilities->data[row * probabilities->cols + col];
            for (size_t other = 0; other <= row; ++other) {
                float upstream = 0.0f;
                for (size_t dimension = 0; dimension < output->cols; ++dimension) {
                    upstream += output->grad[row * output->cols + dimension] *
                        value->data[other * value->cols + dimension];
                }
                softmax_gradient -= probabilities->data[row * probabilities->cols + col] *
                    probabilities->data[row * probabilities->cols + other] * upstream;
            }
            for (size_t dimension = 0; dimension < query->cols; ++dimension) {
                query_grad->grad[row * query->cols + dimension] +=
                    scale * softmax_gradient * key->data[col * key->cols + dimension];
                key_grad->grad[col * key->cols + dimension] +=
                    scale * softmax_gradient * query->data[row * query->cols + dimension];
            }
        }
    }
    return 0;
}

static int valid_multi_head_attention(const Tensor *query, const Tensor *key,
                                      const Tensor *value, size_t batch,
                                      size_t heads, size_t sequence_length,
                                      float scale, const Tensor *probabilities,
                                      const Tensor *output) {
    const size_t rows = batch * heads * sequence_length;
    return is_matrix(query) && is_matrix(key) && is_matrix(value) &&
           is_matrix(probabilities) && is_matrix(output) && batch > 0 &&
           heads > 0 && sequence_length > 0 && scale > 0.0f &&
           query->rows == rows && key->rows == rows && value->rows == rows &&
           query->cols == key->cols && probabilities->rows == rows &&
           probabilities->cols == sequence_length && output->rows == rows;
}

typedef struct {
    const Tensor *query;
    const Tensor *key;
    const Tensor *value;
    size_t sequence_length;
    float scale;
    Tensor *probabilities;
    Tensor *output;
    Tensor *query_grad;
    Tensor *key_grad;
    Tensor *value_grad;
} AttentionContext;

static void attention_forward_groups(void *context_pointer, size_t begin,
                                     size_t end) {
    const AttentionContext *context = context_pointer;
    const size_t sequence_length = context->sequence_length;
    const size_t head_dimension = context->query->cols;
    const size_t value_dimension = context->value->cols;
    for (size_t group = begin; group < end; ++group) {
        const size_t group_start = group * sequence_length;
        for (size_t query_index = 0; query_index < sequence_length;
             ++query_index) {
            const size_t query_row = group_start + query_index;
            const float *query_values =
                context->query->data + query_row * head_dimension;
            float *probability_row =
                context->probabilities->data + query_row * sequence_length;
            float maximum = -INFINITY;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                const float *key_values = context->key->data +
                    (group_start + key_index) * head_dimension;
                float score = 0.0f;
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    score += query_values[dimension] * key_values[dimension];
                }
                score *= context->scale;
                probability_row[key_index] = score;
                if (score > maximum) {
                    maximum = score;
                }
            }
            float denominator = 0.0f;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                probability_row[key_index] =
                    expf(probability_row[key_index] - maximum);
                denominator += probability_row[key_index];
            }
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                probability_row[key_index] /= denominator;
            }
            float *output_values =
                context->output->data + query_row * value_dimension;
            for (size_t dimension = 0; dimension < value_dimension;
                 ++dimension) {
                float result = 0.0f;
                for (size_t key_index = 0; key_index < sequence_length;
                     ++key_index) {
                    result += probability_row[key_index] *
                              context->value->data[
                                  (group_start + key_index) * value_dimension +
                                  dimension];
                }
                output_values[dimension] = result;
            }
        }
    }
}

int ops_multi_head_attention(const Tensor *query, const Tensor *key,
                             const Tensor *value, size_t batch,
                             size_t heads, size_t sequence_length,
                             float scale, Tensor *probabilities,
                             Tensor *output) {
    if (!valid_multi_head_attention(query, key, value, batch, heads,
                                    sequence_length, scale, probabilities,
                                    output) ||
        output->cols != value->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(multi_head_attention, query, key, value, batch, heads,
                      sequence_length, scale, probabilities, output)) {
        return 0;
    }
    ops_synchronize();
    AttentionContext context = {
        query, key, value, sequence_length, scale, probabilities, output,
        NULL, NULL, NULL,
    };
    ops_parallel_for(batch * heads, 1, attention_forward_groups, &context);
    return 0;
}

static void attention_backward_groups(void *context_pointer, size_t begin,
                                      size_t end) {
    const AttentionContext *context = context_pointer;
    const size_t sequence_length = context->sequence_length;
    const size_t head_dimension = context->query->cols;
    const size_t value_dimension = context->value->cols;
    for (size_t group = begin; group < end; ++group) {
        const size_t group_start = group * sequence_length;
        for (size_t query_index = 0; query_index < sequence_length;
             ++query_index) {
            const size_t query_row = group_start + query_index;
            const float *output_gradient =
                context->output->grad + query_row * value_dimension;
            const float *probability_row =
                context->probabilities->data + query_row * sequence_length;
            float probability_dot_gradient = 0.0f;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                const size_t key_row = group_start + key_index;
                float probability_gradient = 0.0f;
                for (size_t dimension = 0; dimension < value_dimension;
                     ++dimension) {
                    probability_gradient += output_gradient[dimension] *
                        context->value->data[key_row * value_dimension +
                                             dimension];
                    context->value_grad->grad[key_row * value_dimension +
                                              dimension] +=
                        probability_row[key_index] * output_gradient[dimension];
                }
                probability_dot_gradient +=
                    probability_row[key_index] * probability_gradient;
            }
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                const size_t key_row = group_start + key_index;
                float probability_gradient = 0.0f;
                for (size_t dimension = 0; dimension < value_dimension;
                     ++dimension) {
                    probability_gradient += output_gradient[dimension] *
                        context->value->data[key_row * value_dimension +
                                             dimension];
                }
                const float score_gradient = context->scale *
                    probability_row[key_index] *
                    (probability_gradient - probability_dot_gradient);
                for (size_t dimension = 0; dimension < head_dimension;
                     ++dimension) {
                    context->query_grad->grad[query_row * head_dimension +
                                              dimension] +=
                        score_gradient *
                        context->key->data[key_row * head_dimension + dimension];
                    context->key_grad->grad[key_row * head_dimension +
                                            dimension] +=
                        score_gradient *
                        context->query->data[query_row * head_dimension +
                                             dimension];
                }
            }
        }
    }
}

int ops_multi_head_attention_backward(
    const Tensor *query, const Tensor *key, const Tensor *value,
    size_t batch, size_t heads, size_t sequence_length, float scale,
    const Tensor *probabilities, const Tensor *output, Tensor *query_grad,
    Tensor *key_grad, Tensor *value_grad) {
    if (!valid_multi_head_attention(query, key, value, batch, heads,
                                    sequence_length, scale, probabilities,
                                    output) ||
        !is_matrix(query_grad) || !is_matrix(key_grad) ||
        !is_matrix(value_grad) ||
        output->cols != value->cols || query_grad->rows != query->rows ||
        query_grad->cols != query->cols || key_grad->rows != key->rows ||
        key_grad->cols != key->cols || value_grad->rows != value->rows ||
        value_grad->cols != value->cols) {
        return -1;
    }
    if (RUN_ON_DEVICE(multi_head_attention_backward, query, key, value, batch,
                      heads, sequence_length, scale, probabilities, output,
                      query_grad, key_grad, value_grad)) {
        return 0;
    }
    ops_synchronize();
    AttentionContext context = {
        query, key, value, sequence_length, scale, (Tensor *)probabilities,
        (Tensor *)output, query_grad, key_grad, value_grad,
    };
    ops_parallel_for(batch * heads, 1, attention_backward_groups, &context);
    return 0;
}

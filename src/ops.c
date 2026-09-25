#include "ops.h"

#include <math.h>

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

    for (size_t i = 0; i < left->rows * left->cols; ++i) {
        left->grad[i] += output->grad[i];
        right->grad[i] += output->grad[i];
    }
    return 0;
}

int ops_bias_add(const Tensor *input, const Tensor *bias, Tensor *output) {
    if (!is_matrix(input) || !is_matrix(bias) || !is_matrix(output) ||
        bias->rows != 1 || bias->cols != input->cols ||
        output->rows != input->rows || output->cols != input->cols) {
        return -1;
    }
    for (size_t row = 0; row < input->rows; ++row) {
        for (size_t col = 0; col < input->cols; ++col) {
            output->data[row * output->cols + col] =
                input->data[row * input->cols + col] + bias->data[col];
        }
    }
    return 0;
}

int ops_bias_add_backward(const Tensor *output, Tensor *bias_grad) {
    if (!is_matrix(output) || !is_matrix(bias_grad) ||
        bias_grad->rows != 1 || bias_grad->cols != output->cols) {
        return -1;
    }
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t col = 0; col < output->cols; ++col) {
            bias_grad->grad[col] += output->grad[row * output->cols + col];
        }
    }
    return 0;
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
    for (size_t row = 0; row < input->rows; ++row) {
        float mean = 0.0f;
        for (size_t col = 0; col < input->cols; ++col) {
            mean += input->data[row * input->cols + col];
        }
        mean /= (float)input->cols;
        float variance = 0.0f;
        for (size_t col = 0; col < input->cols; ++col) {
            const float centered = input->data[row * input->cols + col] - mean;
            variance += centered * centered;
        }
        variance /= (float)input->cols;
        const float inverse_std = 1.0f / sqrtf(variance + epsilon);
        for (size_t col = 0; col < input->cols; ++col) {
            const float normalized =
                (input->data[row * input->cols + col] - mean) * inverse_std;
            output->data[row * output->cols + col] =
                normalized * gamma->data[col] + beta->data[col];
        }
    }
    return 0;
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
    for (size_t row = 0; row < input->rows; ++row) {
        float mean = 0.0f;
        for (size_t col = 0; col < input->cols; ++col) {
            mean += input->data[row * input->cols + col];
        }
        mean /= (float)input->cols;
        float variance = 0.0f;
        for (size_t col = 0; col < input->cols; ++col) {
            const float centered = input->data[row * input->cols + col] - mean;
            variance += centered * centered;
        }
        variance /= (float)input->cols;
        const float inverse_std = 1.0f / sqrtf(variance + epsilon);
        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (size_t col = 0; col < input->cols; ++col) {
            const float centered = input->data[row * input->cols + col] - mean;
            const float normalized = centered * inverse_std;
            const float upstream = output->grad[row * output->cols + col];
            const float dxhat = upstream * gamma->data[col];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * normalized;
            gamma_grad->grad[col] += upstream * normalized;
            beta_grad->grad[col] += upstream;
        }
        for (size_t col = 0; col < input->cols; ++col) {
            const float centered = input->data[row * input->cols + col] - mean;
            const float normalized = centered * inverse_std;
            const float dxhat = output->grad[row * output->cols + col] *
                                gamma->data[col];
            input_grad->grad[row * input->cols + col] +=
                inverse_std * (dxhat - sum_dxhat / (float)input->cols -
                               normalized * sum_dxhat_xhat / (float)input->cols);
        }
    }
    return 0;
}

int ops_gelu(const Tensor *input, Tensor *output) {
    if (!is_matrix(input) || !is_matrix(output) ||
        input->rows != output->rows || input->cols != output->cols) {
        return -1;
    }
    const float coefficient = sqrtf(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        const float x = input->data[i];
        output->data[i] = 0.5f * x *
                          (1.0f + tanhf(coefficient * (x + 0.044715f * x * x * x)));
    }
    return 0;
}

int ops_gelu_backward(const Tensor *input, const Tensor *output) {
    if (!is_matrix(input) || !is_matrix(output) ||
        input->rows != output->rows || input->cols != output->cols) {
        return -1;
    }
    const float coefficient = sqrtf(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        const float x = input->data[i];
        const float inner = coefficient * (x + 0.044715f * x * x * x);
        const float tanh_inner = tanhf(inner);
        const float derivative = 0.5f * (1.0f + tanh_inner) +
                                 0.5f * x * (1.0f - tanh_inner * tanh_inner) *
                                 coefficient * (1.0f + 3.0f * 0.044715f * x * x);
        ((Tensor *)input)->grad[i] += output->grad[i] * derivative;
    }
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
    if (!is_matrix(logits) || !targets || !loss || !is_matrix(logits_grad) ||
        logits_grad->rows != logits->rows || logits_grad->cols != logits->cols) {
        return -1;
    }
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
        const float target_logit = logits->data[row * logits->cols + targets[row]];
        const float weight = class_weights ? class_weights[targets[row]] : 1.0f;
        *loss += weight * (-target_logit + maximum + logf(denominator));
        for (size_t col = 0; col < logits->cols; ++col) {
            const float probability =
                expf(logits->data[row * logits->cols + col] - maximum) / denominator;
            logits_grad->grad[row * logits->cols + col] +=
                weight * (probability - (col == targets[row] ? 1.0f : 0.0f));
        }
    }
    *loss /= (float)logits->rows;
    for (size_t i = 0; i < logits->rows * logits->cols; ++i) {
        logits_grad->grad[i] /= (float)logits->rows;
    }
    return 0;
}

int ops_adamw_step(Parameter *parameter, float learning_rate,
                   float beta1, float beta2, float epsilon,
                   float weight_decay) {
    if (!parameter || !parameter->value.data || !parameter->value.grad ||
        !parameter->m || !parameter->v || learning_rate <= 0.0f ||
        beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f ||
        epsilon <= 0.0f || weight_decay < 0.0f) {
        return -1;
    }
    parameter->step += 1;
    const size_t count = tensor_numel(&parameter->value);
    const float correction1 = 1.0f - powf(beta1, (float)parameter->step);
    const float correction2 = 1.0f - powf(beta2, (float)parameter->step);
    for (size_t i = 0; i < count; ++i) {
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
    return 0;
}

int ops_causal_attention(const Tensor *query, const Tensor *key,
                         const Tensor *value, float scale,
                         Tensor *probabilities, Tensor *output) {
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
    const size_t rows_per_head = sequence_length;
    const size_t groups = batch * heads;
    for (size_t group = 0; group < groups; ++group) {
        const size_t group_start = group * rows_per_head;
        for (size_t query_index = 0; query_index < sequence_length;
             ++query_index) {
            const size_t query_row = group_start + query_index;
            float maximum = -INFINITY;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                float score = 0.0f;
                for (size_t dimension = 0; dimension < query->cols; ++dimension) {
                    score += query->data[query_row * query->cols + dimension] *
                             key->data[(group_start + key_index) * key->cols +
                                       dimension];
                }
                score *= scale;
                probabilities->data[query_row * probabilities->cols + key_index] =
                    score;
                if (score > maximum) {
                    maximum = score;
                }
            }
            float denominator = 0.0f;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                float *probability =
                    &probabilities->data[query_row * probabilities->cols + key_index];
                *probability = expf(*probability - maximum);
                denominator += *probability;
            }
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                probabilities->data[query_row * probabilities->cols + key_index] /=
                    denominator;
            }
            for (size_t dimension = 0; dimension < output->cols; ++dimension) {
                float result = 0.0f;
                for (size_t key_index = 0; key_index < sequence_length;
                     ++key_index) {
                    result += probabilities->data[
                                  query_row * probabilities->cols + key_index] *
                              value->data[(group_start + key_index) * value->cols +
                                          dimension];
                }
                output->data[query_row * output->cols + dimension] = result;
            }
        }
    }
    return 0;
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
    const size_t groups = batch * heads;
    for (size_t group = 0; group < groups; ++group) {
        const size_t group_start = group * sequence_length;
        for (size_t query_index = 0; query_index < sequence_length;
             ++query_index) {
            const size_t query_row = group_start + query_index;
            float probability_dot_gradient = 0.0f;
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                const size_t key_row = group_start + key_index;
                float probability_gradient = 0.0f;
                for (size_t dimension = 0; dimension < output->cols; ++dimension) {
                    probability_gradient +=
                        output->grad[query_row * output->cols + dimension] *
                        value->data[key_row * value->cols + dimension];
                    value_grad->grad[key_row * value->cols + dimension] +=
                        probabilities->data[query_row * probabilities->cols +
                                             key_index] *
                        output->grad[query_row * output->cols + dimension];
                }
                probability_dot_gradient +=
                    probabilities->data[query_row * probabilities->cols + key_index] *
                    probability_gradient;
            }
            for (size_t key_index = 0; key_index < sequence_length;
                 ++key_index) {
                const size_t key_row = group_start + key_index;
                float probability_gradient = 0.0f;
                for (size_t dimension = 0; dimension < output->cols; ++dimension) {
                    probability_gradient +=
                        output->grad[query_row * output->cols + dimension] *
                        value->data[key_row * value->cols + dimension];
                }
                const float score_gradient =
                    probabilities->data[query_row * probabilities->cols + key_index] *
                    (probability_gradient - probability_dot_gradient);
                for (size_t dimension = 0; dimension < query->cols; ++dimension) {
                    query_grad->grad[query_row * query->cols + dimension] +=
                        scale * score_gradient *
                        key->data[key_row * key->cols + dimension];
                    key_grad->grad[key_row * key->cols + dimension] +=
                        scale * score_gradient *
                        query->data[query_row * query->cols + dimension];
                }
            }
        }
    }
    return 0;
}

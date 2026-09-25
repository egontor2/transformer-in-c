#include "ops.h"

#include <math.h>

static int is_matrix(const Tensor *tensor) {
    return tensor && tensor->ndim == 2 && tensor->data && tensor->grad;
}

int ops_gemm(const Tensor *left, const Tensor *right, Tensor *output) {
    if (!is_matrix(left) || !is_matrix(right) || !is_matrix(output) ||
        left->cols != right->rows || output->rows != left->rows ||
        output->cols != right->cols) {
        return -1;
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
        *loss += -target_logit + maximum + logf(denominator);
        for (size_t col = 0; col < logits->cols; ++col) {
            const float probability =
                expf(logits->data[row * logits->cols + col] - maximum) / denominator;
            logits_grad->grad[row * logits->cols + col] +=
                probability - (col == targets[row] ? 1.0f : 0.0f);
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

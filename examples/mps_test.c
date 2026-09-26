#include "mps_backend.h"
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static float pseudo_random_value(size_t index, float frequency) {
    return sinf((float)index * frequency) + 0.25f * cosf((float)index * 0.37f);
}

static int fill_tensor_pair(Tensor *reference, Tensor *accelerated,
                            size_t rows, size_t cols, float frequency) {
    if (tensor_init(reference, rows, cols) != 0 ||
        tensor_init(accelerated, rows, cols) != 0) {
        return -1;
    }
    for (size_t i = 0; i < rows * cols; ++i) {
        reference->data[i] = accelerated->data[i] =
            pseudo_random_value(i, frequency);
        reference->grad[i] = accelerated->grad[i] =
            pseudo_random_value(i, frequency * 0.5f);
    }
    return 0;
}

static float max_abs_difference(const float *left, const float *right,
                                size_t count) {
    float difference = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        difference = fmaxf(difference, fabsf(left[i] - right[i]));
    }
    return difference;
}

static int check_rectangular_gemm(MPSBackend *backend, size_t left_rows,
                                  size_t left_cols, size_t right_cols) {
    Tensor reference_left = {0}, accelerated_left = {0};
    Tensor reference_right = {0}, accelerated_right = {0};
    Tensor reference_output = {0}, accelerated_output = {0};
    int failed =
        fill_tensor_pair(&reference_left, &accelerated_left, left_rows,
                         left_cols, 0.11f) ||
        fill_tensor_pair(&reference_right, &accelerated_right, left_cols,
                         right_cols, 0.23f) ||
        fill_tensor_pair(&reference_output, &accelerated_output, left_rows,
                         right_cols, 0.07f);
    const float tolerance = 1e-3f;
    if (!failed) {
        ops_reset_gemm_backend();
        failed = ops_gemm(&reference_left, &reference_right,
                          &reference_output) != 0 ||
            ops_gemm_backward(&reference_left, &reference_right,
                              &reference_output) != 0;
        ops_set_gemm_backend(mps_backend_gemm_callback, backend);
        ops_set_gemm_backward_backend(mps_backend_gemm_backward_callback,
                                      backend);
        failed = failed ||
            ops_gemm(&accelerated_left, &accelerated_right,
                     &accelerated_output) != 0 ||
            ops_gemm_backward(&accelerated_left, &accelerated_right,
                              &accelerated_output) != 0;
    }
    if (!failed) {
        const float output_error = max_abs_difference(
            reference_output.data, accelerated_output.data,
            left_rows * right_cols);
        const float left_gradient_error = max_abs_difference(
            reference_left.grad, accelerated_left.grad, left_rows * left_cols);
        const float right_gradient_error = max_abs_difference(
            reference_right.grad, accelerated_right.grad,
            left_cols * right_cols);
        failed = output_error > tolerance ||
                 left_gradient_error > tolerance ||
                 right_gradient_error > tolerance;
        printf("MPS GEMM %zux%zu * %zux%zu: output=%.2e left_grad=%.2e "
               "right_grad=%.2e %s\n",
               left_rows, left_cols, left_cols, right_cols, output_error,
               left_gradient_error, right_gradient_error,
               failed ? "failed" : "ok");
    }
    tensor_free(&reference_left);
    tensor_free(&accelerated_left);
    tensor_free(&reference_right);
    tensor_free(&accelerated_right);
    tensor_free(&reference_output);
    tensor_free(&accelerated_output);
    return failed;
}

int main(void) {
    MPSBackend *backend = NULL;
    if (mps_backend_create(&backend) != 0) {
        fprintf(stderr, "MPS backend unavailable\n");
        return 1;
    }
    const float left[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float right[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float output[4] = {0};
    ops_set_gemm_backend(mps_backend_gemm_callback, backend);
    ops_set_gemm_backward_backend(mps_backend_gemm_backward_callback, backend);
    Tensor left_tensor = {0};
    Tensor right_tensor = {0};
    Tensor output_tensor = {0};
    const int initialized =
        tensor_init(&left_tensor, 2, 2) ||
        tensor_init(&right_tensor, 2, 2) ||
        tensor_init(&output_tensor, 2, 2);
    if (initialized == 0) {
        memcpy(left_tensor.data, left, sizeof(left));
        memcpy(right_tensor.data, right, sizeof(right));
    }
    int result = initialized == 0
                     ? ops_gemm(&left_tensor, &right_tensor, &output_tensor)
                     : -1;
    if (result == 0) {
        for (int iteration = 0; iteration < 3; ++iteration) {
            if (ops_gemm(&left_tensor, &right_tensor, &output_tensor) != 0) {
                result = -1;
                break;
            }
        }
    }
    if (result == 0) {
        memcpy(output, output_tensor.data, sizeof(output));
        for (size_t i = 0; i < 4; ++i) {
            output_tensor.grad[i] = 1.0f;
        }
        if (ops_gemm_backward(&left_tensor, &right_tensor,
                              &output_tensor) != 0) {
            result = -1;
        }
    }
    const int valid = result == 0 && fabsf(output[0] - 19.0f) < 1e-5f &&
                      fabsf(output[1] - 22.0f) < 1e-5f &&
                      fabsf(output[2] - 43.0f) < 1e-5f &&
                      fabsf(output[3] - 50.0f) < 1e-5f &&
                      fabsf(left_tensor.grad[0] - 11.0f) < 1e-5f &&
                      fabsf(left_tensor.grad[1] - 15.0f) < 1e-5f &&
                      fabsf(left_tensor.grad[2] - 11.0f) < 1e-5f &&
                      fabsf(left_tensor.grad[3] - 15.0f) < 1e-5f &&
                      fabsf(right_tensor.grad[0] - 4.0f) < 1e-5f &&
                      fabsf(right_tensor.grad[1] - 4.0f) < 1e-5f &&
                      fabsf(right_tensor.grad[2] - 6.0f) < 1e-5f &&
                      fabsf(right_tensor.grad[3] - 6.0f) < 1e-5f;
    printf("MPS GEMM: %s\n", valid ? "ok" : "failed");
    const size_t rectangular_shapes[][3] = {
        {3, 5, 7}, {50, 128, 384}, {1600, 128, 512}, {1600, 512, 128},
    };
    int rectangular_failures = 0;
    for (size_t i = 0; i < 4; ++i) {
        for (int repetition = 0; repetition < 2; ++repetition) {
            rectangular_failures += check_rectangular_gemm(
                backend, rectangular_shapes[i][0], rectangular_shapes[i][1],
                rectangular_shapes[i][2]);
        }
    }
    ops_reset_gemm_backend();
    tensor_free(&left_tensor);
    tensor_free(&right_tensor);
    tensor_free(&output_tensor);
    mps_backend_free(backend);
    return valid && rectangular_failures == 0 ? 0 : 1;
}

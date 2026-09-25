#include "mps_backend.h"
#include "ops.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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
    ops_reset_gemm_backend();
    tensor_free(&left_tensor);
    tensor_free(&right_tensor);
    tensor_free(&output_tensor);
    mps_backend_free(backend);
    return valid ? 0 : 1;
}

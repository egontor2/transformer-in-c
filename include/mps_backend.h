#ifndef MPS_BACKEND_H
#define MPS_BACKEND_H

#include <stddef.h>

typedef struct MPSBackend MPSBackend;

#ifdef __cplusplus
extern "C" {
#endif

int mps_backend_create(MPSBackend **backend);
void mps_backend_free(MPSBackend *backend);
int mps_backend_available(const MPSBackend *backend);
int mps_backend_gemm(MPSBackend *backend, const float *left,
                     const float *right, float *output, size_t left_rows,
                     size_t left_cols, size_t right_cols);
int mps_backend_gemm_callback(void *context, const float *left,
                              const float *right, float *output,
                              size_t left_rows, size_t left_cols,
                              size_t right_cols);
int mps_backend_gemm_backward_callback(
    void *context, const float *left, const float *right,
    const float *output_grad, float *left_grad, float *right_grad,
    size_t left_rows, size_t left_cols, size_t right_cols);

#ifdef __cplusplus
}
#endif

#endif

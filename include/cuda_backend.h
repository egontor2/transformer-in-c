#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include <stddef.h>

typedef struct CudaBackend CudaBackend;

#ifdef __cplusplus
extern "C" {
#endif

int cuda_backend_create(CudaBackend **backend);
void cuda_backend_free(CudaBackend *backend);
int cuda_backend_available(const CudaBackend *backend);
int cuda_backend_enable_device_execution(CudaBackend *backend);
void cuda_backend_disable_device_execution(CudaBackend *backend);

#ifdef __cplusplus
}
#endif

#endif

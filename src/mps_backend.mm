#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct MPSBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

static id<MTLBuffer> create_buffer(id<MTLDevice> device, const float *data,
                                   size_t count) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    if (buffer && data) {
        memcpy(buffer.contents, data, count * sizeof(float));
    }
    return buffer;
}

int mps_backend_create(MPSBackend **backend) {
    if (!backend) {
        return -1;
    }
    *backend = NULL;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return 1;
    }
    MPSBackend *result = (MPSBackend *)calloc(1, sizeof(*result));
    if (!result) {
        return -1;
    }
    result->device = device;
    result->queue = [device newCommandQueue];
    if (!result->queue) {
        free(result);
        return -1;
    }
    *backend = result;
    return 0;
}

void mps_backend_free(MPSBackend *backend) {
    free(backend);
}

int mps_backend_available(const MPSBackend *backend) {
    return backend && backend->device && backend->queue ? 1 : 0;
}

int mps_backend_gemm(MPSBackend *backend, const float *left,
                     const float *right, float *output, size_t left_rows,
                     size_t left_cols, size_t right_cols) {
    if (!mps_backend_available(backend) || !left || !right || !output ||
        left_rows == 0 || left_cols == 0 || right_cols == 0) {
        return -1;
    }
    if (left_rows > SIZE_MAX / left_cols ||
        left_cols > SIZE_MAX / right_cols ||
        left_rows > SIZE_MAX / right_cols) {
        return -1;
    }
    const size_t left_count = left_rows * left_cols;
    const size_t right_count = left_cols * right_cols;
    const size_t output_count = left_rows * right_cols;
    id<MTLBuffer> left_buffer = create_buffer(backend->device, left, left_count);
    id<MTLBuffer> right_buffer =
        create_buffer(backend->device, right, right_count);
    id<MTLBuffer> output_buffer =
        create_buffer(backend->device, NULL, output_count);
    if (!left_buffer || !right_buffer || !output_buffer) {
        return -1;
    }
    MPSMatrixDescriptor *left_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                               columns:left_cols
                                              rowBytes:left_cols * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *right_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:left_cols
                                               columns:right_cols
                                              rowBytes:right_cols * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                               columns:right_cols
                                              rowBytes:right_cols * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrix *left_matrix =
        [[MPSMatrix alloc] initWithBuffer:left_buffer descriptor:left_descriptor];
    MPSMatrix *right_matrix =
        [[MPSMatrix alloc] initWithBuffer:right_buffer
                                  descriptor:right_descriptor];
    MPSMatrix *output_matrix =
        [[MPSMatrix alloc] initWithBuffer:output_buffer
                                  descriptor:output_descriptor];
    MPSMatrixMultiplication *multiplication =
        [[MPSMatrixMultiplication alloc] initWithDevice:backend->device
                                         transposeLeft:NO
                                        transposeRight:NO
                                           resultRows:left_rows
                                        resultColumns:right_cols
                                      interiorColumns:left_cols
                                                 alpha:1.0
                                                  beta:0.0];
    id<MTLCommandBuffer> command_buffer = [backend->queue commandBuffer];
    if (!left_matrix || !right_matrix || !output_matrix || !multiplication ||
        !command_buffer) {
        return -1;
    }
    [multiplication encodeToCommandBuffer:command_buffer
                              leftMatrix:left_matrix
                             rightMatrix:right_matrix
                            resultMatrix:output_matrix];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        return -1;
    }
    memcpy(output, output_buffer.contents, output_count * sizeof(float));
    return 0;
}

int mps_backend_gemm_callback(void *context, const float *left,
                              const float *right, float *output,
                              size_t left_rows, size_t left_cols,
                              size_t right_cols) {
    return mps_backend_gemm((MPSBackend *)context, left, right, output,
                            left_rows, left_cols, right_cols);
}

int mps_backend_gemm_backward_callback(
    void *context, const float *left, const float *right,
    const float *output_grad, float *left_grad, float *right_grad,
    size_t left_rows, size_t left_cols, size_t right_cols) {
    if (!context || !left || !right || !output_grad || !left_grad ||
        !right_grad || left_rows == 0 || left_cols == 0 || right_cols == 0) {
        return -1;
    }
    float *right_transposed =
        (float *)malloc(left_cols * right_cols * sizeof(float));
    float *left_transposed =
        (float *)malloc(left_rows * left_cols * sizeof(float));
    float *left_update =
        (float *)malloc(left_rows * left_cols * sizeof(float));
    float *right_update =
        (float *)malloc(left_cols * right_cols * sizeof(float));
    if (!right_transposed || !left_transposed || !left_update ||
        !right_update) {
        free(right_transposed);
        free(left_transposed);
        free(left_update);
        free(right_update);
        return -1;
    }
    for (size_t row = 0; row < right_cols; ++row) {
        for (size_t col = 0; col < left_cols; ++col) {
            right_transposed[row * left_cols + col] =
                right[col * right_cols + row];
        }
    }
    for (size_t row = 0; row < left_rows; ++row) {
        for (size_t col = 0; col < left_cols; ++col) {
            left_transposed[col * left_rows + row] =
                left[row * left_cols + col];
        }
    }
    const int left_result =
        mps_backend_gemm((MPSBackend *)context, output_grad,
                         right_transposed, left_update, left_rows, right_cols,
                         left_cols);
    const int right_result =
        mps_backend_gemm((MPSBackend *)context, left_transposed, output_grad,
                         right_update, left_cols, left_rows, right_cols);
    if (left_result == 0 && right_result == 0) {
        for (size_t i = 0; i < left_rows * left_cols; ++i) {
            left_grad[i] += left_update[i];
        }
        for (size_t i = 0; i < left_cols * right_cols; ++i) {
            right_grad[i] += right_update[i];
        }
    }
    free(right_transposed);
    free(left_transposed);
    free(left_update);
    free(right_update);
    return left_result == 0 && right_result == 0 ? 0 : -1;
}

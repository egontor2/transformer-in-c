#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static const NSUInteger kMaxCachedGemmShapes = 16;

@interface MPSGEMMBufferCacheEntry : NSObject
@property(nonatomic) size_t leftRows;
@property(nonatomic) size_t leftCols;
@property(nonatomic) size_t rightCols;
@property(nonatomic, strong) id<MTLBuffer> leftBuffer;
@property(nonatomic, strong) id<MTLBuffer> rightBuffer;
@property(nonatomic, strong) id<MTLBuffer> outputBuffer;
@end

@implementation MPSGEMMBufferCacheEntry
@end

struct MPSBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableArray<MPSGEMMBufferCacheEntry *> *bufferCache;
};

static int checked_product(size_t a, size_t b, size_t *result) {
    if (!result || (a != 0 && b > SIZE_MAX / a)) {
        return -1;
    }
    *result = a * b;
    return 0;
}

static id<MTLBuffer> create_buffer(id<MTLDevice> device, const float *data,
                                   size_t count) {
    size_t bytes = 0;
    if (!device || checked_product(count, sizeof(float), &bytes) != 0 ||
        bytes > NSUIntegerMax) {
        return nil;
    }
    id<MTLBuffer> buffer =
        [device newBufferWithLength:(NSUInteger)bytes
                            options:MTLResourceStorageModeShared];
    if (buffer && data && bytes != 0) {
        memcpy(buffer.contents, data, bytes);
    }
    return buffer;
}

static MPSGEMMBufferCacheEntry *find_cached_entry(
    MPSBackend *backend, size_t left_rows, size_t left_cols,
    size_t right_cols) {
    for (MPSGEMMBufferCacheEntry *entry in backend->bufferCache) {
        if (entry.leftRows == left_rows && entry.leftCols == left_cols &&
            entry.rightCols == right_cols) {
            return entry;
        }
    }
    return nil;
}

static MPSGEMMBufferCacheEntry *get_cached_entry(
    MPSBackend *backend, size_t left_rows, size_t left_cols,
    size_t right_cols, size_t left_count, size_t right_count,
    size_t output_count) {
    MPSGEMMBufferCacheEntry *entry =
        find_cached_entry(backend, left_rows, left_cols, right_cols);
    if (entry) {
        return entry;
    }
    if (backend->bufferCache.count >= kMaxCachedGemmShapes) {
        [backend->bufferCache removeObjectAtIndex:0];
    }
    entry = [[MPSGEMMBufferCacheEntry alloc] init];
    entry.leftRows = left_rows;
    entry.leftCols = left_cols;
    entry.rightCols = right_cols;
    entry.leftBuffer = create_buffer(backend->device, NULL, left_count);
    entry.rightBuffer = create_buffer(backend->device, NULL, right_count);
    entry.outputBuffer = create_buffer(backend->device, NULL, output_count);
    if (!entry.leftBuffer || !entry.rightBuffer || !entry.outputBuffer) {
        return nil;
    }
    [backend->bufferCache addObject:entry];
    return entry;
}

int mps_backend_create(MPSBackend **backend) {
    int status = -1;
    @autoreleasepool {
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
            result->device = nil;
            free(result);
            return -1;
        }
        result->bufferCache = [[NSMutableArray alloc] init];
        if (!result->bufferCache) {
            result->queue = nil;
            result->device = nil;
            free(result);
            return -1;
        }
        *backend = result;
        status = 0;
    }
    return status;
}

void mps_backend_free(MPSBackend *backend) {
    if (!backend) {
        return;
    }
    backend->bufferCache = nil;
    backend->queue = nil;
    backend->device = nil;
    free(backend);
}

int mps_backend_available(const MPSBackend *backend) {
    return backend && backend->device && backend->queue ? 1 : 0;
}

static int mps_backend_gemm_impl(MPSBackend *backend, const float *left,
                                  const float *right, float *output,
                                  size_t left_rows, size_t left_cols,
                                  size_t right_cols) {
    if (!mps_backend_available(backend) || !left || !right || !output ||
        left_rows == 0 || left_cols == 0 || right_cols == 0) {
        return -1;
    }
    size_t left_count = 0;
    size_t right_count = 0;
    size_t output_count = 0;
    if (checked_product(left_rows, left_cols, &left_count) != 0 ||
        checked_product(left_cols, right_cols, &right_count) != 0 ||
        checked_product(left_rows, right_cols, &output_count) != 0) {
        return -1;
    }
    MPSGEMMBufferCacheEntry *entry =
        get_cached_entry(backend, left_rows, left_cols, right_cols,
                         left_count, right_count, output_count);
    if (!entry) {
        return -1;
    }
    const size_t left_bytes = left_count * sizeof(float);
    const size_t right_bytes = right_count * sizeof(float);
    memcpy(entry.leftBuffer.contents, left, left_bytes);
    memcpy(entry.rightBuffer.contents, right, right_bytes);
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
        [[MPSMatrix alloc] initWithBuffer:entry.leftBuffer
                                descriptor:left_descriptor];
    MPSMatrix *right_matrix =
        [[MPSMatrix alloc] initWithBuffer:entry.rightBuffer
                                  descriptor:right_descriptor];
    MPSMatrix *output_matrix =
        [[MPSMatrix alloc] initWithBuffer:entry.outputBuffer
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
    memcpy(output, entry.outputBuffer.contents, output_count * sizeof(float));
    return 0;
}

int mps_backend_gemm(MPSBackend *backend, const float *left,
                     const float *right, float *output, size_t left_rows,
                     size_t left_cols, size_t right_cols) {
    int result = -1;
    @autoreleasepool {
        result = mps_backend_gemm_impl(backend, left, right, output,
                                       left_rows, left_cols, right_cols);
    }
    return result;
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
    size_t left_count = 0;
    size_t right_count = 0;
    if (checked_product(left_rows, left_cols, &left_count) != 0 ||
        checked_product(left_cols, right_cols, &right_count) != 0) {
        return -1;
    }
    size_t left_bytes = 0;
    size_t right_bytes = 0;
    if (checked_product(left_count, sizeof(float), &left_bytes) != 0 ||
        checked_product(right_count, sizeof(float), &right_bytes) != 0) {
        return -1;
    }
    float *right_transposed = (float *)malloc(right_bytes);
    float *left_transposed = (float *)malloc(left_bytes);
    float *left_update = (float *)malloc(left_bytes);
    float *right_update = (float *)malloc(right_bytes);
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

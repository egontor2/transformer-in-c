#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.h"

extern "C" {
#include "ops.h"
#include "tensor.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "metal_kernels.inc"

static const NSUInteger kMaxCachedGemmPlans = 32;

typedef NS_ENUM(NSInteger, MPSGEMMPlanKind) {
    MPSGEMMPlanKindForward,
    MPSGEMMPlanKindBackward,
};

@interface MPSGEMMPlan : NSObject
@property(nonatomic) MPSGEMMPlanKind kind;
@property(nonatomic) size_t leftRows;
@property(nonatomic) size_t leftCols;
@property(nonatomic) size_t rightCols;
@property(nonatomic, strong) MPSMatrix *leftMatrix;
@property(nonatomic, strong) MPSMatrix *rightMatrix;
@property(nonatomic, strong) MPSMatrix *outputMatrix;
@property(nonatomic, strong) MPSMatrix *leftGradientMatrix;
@property(nonatomic, strong) MPSMatrix *rightGradientMatrix;
@property(nonatomic, strong) MPSMatrixMultiplication *forwardProduct;
@property(nonatomic, strong) MPSMatrixMultiplication *leftGradientProduct;
@property(nonatomic, strong) MPSMatrixMultiplication *rightGradientProduct;
@end

@implementation MPSGEMMPlan
@end

struct MPSBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableArray<MPSGEMMPlan *> *planCache;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *tensorProducts;
    id<MTLCommandBuffer> pendingCommandBuffer;
    id<MTLComputeCommandEncoder> computeEncoder;
    id<MTLBuffer> statisticsBuffer;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *scratchBuffers;
    OpsDeviceBackend deviceBackend;
};

static int checked_product(size_t a, size_t b, size_t *result) {
    if (!result || (a != 0 && b > SIZE_MAX / a)) {
        return -1;
    }
    *result = a * b;
    return 0;
}

static MPSMatrix *create_matrix(id<MTLDevice> device, size_t rows,
                                size_t cols) {
    size_t count = 0;
    size_t bytes = 0;
    if (checked_product(rows, cols, &count) != 0 ||
        checked_product(count, sizeof(float), &bytes) != 0 ||
        bytes > NSUIntegerMax) {
        return nil;
    }
    id<MTLBuffer> buffer =
        [device newBufferWithLength:(NSUInteger)bytes
                            options:MTLResourceStorageModeShared];
    if (!buffer) {
        return nil;
    }
    MPSMatrixDescriptor *descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                               columns:cols
                                              rowBytes:cols * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}

static MPSMatrixMultiplication *create_product(id<MTLDevice> device,
                                               BOOL transpose_left,
                                               BOOL transpose_right,
                                               size_t result_rows,
                                               size_t result_cols,
                                               size_t interior_cols,
                                               double beta) {
    return [[MPSMatrixMultiplication alloc] initWithDevice:device
                                             transposeLeft:transpose_left
                                            transposeRight:transpose_right
                                                resultRows:result_rows
                                             resultColumns:result_cols
                                           interiorColumns:interior_cols
                                                     alpha:1.0
                                                      beta:beta];
}

static float *matrix_contents(MPSMatrix *matrix) {
    return (float *)matrix.data.contents;
}

static size_t matrix_bytes(MPSMatrix *matrix) {
    return matrix.rows * matrix.columns * sizeof(float);
}

static MPSGEMMPlan *create_plan(MPSBackend *backend, MPSGEMMPlanKind kind,
                                size_t left_rows, size_t left_cols,
                                size_t right_cols) {
    id<MTLDevice> device = backend->device;
    MPSGEMMPlan *plan = [[MPSGEMMPlan alloc] init];
    plan.kind = kind;
    plan.leftRows = left_rows;
    plan.leftCols = left_cols;
    plan.rightCols = right_cols;
    plan.leftMatrix = create_matrix(device, left_rows, left_cols);
    plan.rightMatrix = create_matrix(device, left_cols, right_cols);
    plan.outputMatrix = create_matrix(device, left_rows, right_cols);
    if (!plan.leftMatrix || !plan.rightMatrix || !plan.outputMatrix) {
        return nil;
    }
    if (kind == MPSGEMMPlanKindForward) {
        plan.forwardProduct = create_product(device, NO, NO, left_rows,
                                             right_cols, left_cols, 0.0);
        return plan.forwardProduct ? plan : nil;
    }
    const double accumulate_into_result = 1.0;
    plan.leftGradientMatrix = create_matrix(device, left_rows, left_cols);
    plan.rightGradientMatrix = create_matrix(device, left_cols, right_cols);
    plan.leftGradientProduct =
        create_product(device, NO, YES, left_rows, left_cols, right_cols,
                       accumulate_into_result);
    plan.rightGradientProduct =
        create_product(device, YES, NO, left_cols, right_cols, left_rows,
                       accumulate_into_result);
    return plan.leftGradientMatrix && plan.rightGradientMatrix &&
                   plan.leftGradientProduct && plan.rightGradientProduct
               ? plan
               : nil;
}

static MPSGEMMPlan *get_plan(MPSBackend *backend, MPSGEMMPlanKind kind,
                             size_t left_rows, size_t left_cols,
                             size_t right_cols) {
    const NSUInteger plan_count = backend->planCache.count;
    for (NSUInteger index = 0; index < plan_count; ++index) {
        MPSGEMMPlan *cached_plan = backend->planCache[index];
        if (cached_plan.kind == kind && cached_plan.leftRows == left_rows &&
            cached_plan.leftCols == left_cols &&
            cached_plan.rightCols == right_cols) {
            if (index + 1 != plan_count) {
                [backend->planCache removeObjectAtIndex:index];
                [backend->planCache addObject:cached_plan];
            }
            return cached_plan;
        }
    }
    MPSGEMMPlan *plan =
        create_plan(backend, kind, left_rows, left_cols, right_cols);
    if (!plan) {
        return nil;
    }
    if (backend->planCache.count >= kMaxCachedGemmPlans) {
        [backend->planCache removeObjectAtIndex:0];
    }
    [backend->planCache addObject:plan];
    return plan;
}

static int run_and_wait(id<MTLCommandBuffer> command_buffer) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return command_buffer.status == MTLCommandBufferStatusCompleted ? 0 : -1;
}

static int valid_gemm_shape(size_t left_rows, size_t left_cols,
                            size_t right_cols) {
    size_t count = 0;
    return left_rows != 0 && left_cols != 0 && right_cols != 0 &&
           checked_product(left_rows, left_cols, &count) == 0 &&
           checked_product(left_cols, right_cols, &count) == 0 &&
           checked_product(left_rows, right_cols, &count) == 0;
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
        result->planCache = [[NSMutableArray alloc] init];
        if (!result->planCache) {
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
    if (backend->deviceBackend.context) {
        mps_backend_disable_device_execution(backend);
    }
    backend->planCache = nil;
    backend->pipelines = nil;
    backend->tensorProducts = nil;
    backend->statisticsBuffer = nil;
    backend->scratchBuffers = nil;
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
        !valid_gemm_shape(left_rows, left_cols, right_cols)) {
        return -1;
    }
    MPSGEMMPlan *plan = get_plan(backend, MPSGEMMPlanKindForward, left_rows,
                                 left_cols, right_cols);
    id<MTLCommandBuffer> command_buffer = [backend->queue commandBuffer];
    if (!plan || !command_buffer) {
        return -1;
    }
    memcpy(matrix_contents(plan.leftMatrix), left,
           matrix_bytes(plan.leftMatrix));
    memcpy(matrix_contents(plan.rightMatrix), right,
           matrix_bytes(plan.rightMatrix));
    [plan.forwardProduct encodeToCommandBuffer:command_buffer
                                    leftMatrix:plan.leftMatrix
                                   rightMatrix:plan.rightMatrix
                                  resultMatrix:plan.outputMatrix];
    if (run_and_wait(command_buffer) != 0) {
        return -1;
    }
    memcpy(output, matrix_contents(plan.outputMatrix),
           matrix_bytes(plan.outputMatrix));
    return 0;
}

static int mps_backend_gemm_backward_impl(
    MPSBackend *backend, const float *left, const float *right,
    const float *output_grad, float *left_grad, float *right_grad,
    size_t left_rows, size_t left_cols, size_t right_cols) {
    if (!mps_backend_available(backend) || !left || !right || !output_grad ||
        !left_grad || !right_grad ||
        !valid_gemm_shape(left_rows, left_cols, right_cols)) {
        return -1;
    }
    MPSGEMMPlan *plan = get_plan(backend, MPSGEMMPlanKindBackward, left_rows,
                                 left_cols, right_cols);
    id<MTLCommandBuffer> command_buffer = [backend->queue commandBuffer];
    if (!plan || !command_buffer) {
        return -1;
    }
    memcpy(matrix_contents(plan.leftMatrix), left,
           matrix_bytes(plan.leftMatrix));
    memcpy(matrix_contents(plan.rightMatrix), right,
           matrix_bytes(plan.rightMatrix));
    memcpy(matrix_contents(plan.outputMatrix), output_grad,
           matrix_bytes(plan.outputMatrix));
    memcpy(matrix_contents(plan.leftGradientMatrix), left_grad,
           matrix_bytes(plan.leftGradientMatrix));
    memcpy(matrix_contents(plan.rightGradientMatrix), right_grad,
           matrix_bytes(plan.rightGradientMatrix));
    [plan.leftGradientProduct encodeToCommandBuffer:command_buffer
                                         leftMatrix:plan.outputMatrix
                                        rightMatrix:plan.rightMatrix
                                       resultMatrix:plan.leftGradientMatrix];
    [plan.rightGradientProduct encodeToCommandBuffer:command_buffer
                                          leftMatrix:plan.leftMatrix
                                         rightMatrix:plan.outputMatrix
                                        resultMatrix:plan.rightGradientMatrix];
    if (run_and_wait(command_buffer) != 0) {
        return -1;
    }
    memcpy(left_grad, matrix_contents(plan.leftGradientMatrix),
           matrix_bytes(plan.leftGradientMatrix));
    memcpy(right_grad, matrix_contents(plan.rightGradientMatrix),
           matrix_bytes(plan.rightGradientMatrix));
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
    int result = -1;
    @autoreleasepool {
        result = mps_backend_gemm_backward_impl(
            (MPSBackend *)context, left, right, output_grad, left_grad,
            right_grad, left_rows, left_cols, right_cols);
    }
    return result;
}

static NSMutableDictionary<NSValue *, id<MTLBuffer>> *registered_buffers(void) {
    static NSMutableDictionary<NSValue *, id<MTLBuffer>> *buffers;
    if (!buffers) {
        buffers = [[NSMutableDictionary alloc] init];
    }
    return buffers;
}

static void *allocate_device_memory(void *context, size_t bytes) {
    void *memory = NULL;
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        if (!backend || bytes == 0 || bytes > NSUIntegerMax) {
            return NULL;
        }
        id<MTLBuffer> buffer =
            [backend->device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (buffer) {
            memory = buffer.contents;
            registered_buffers()[[NSValue valueWithPointer:memory]] = buffer;
        }
    }
    return memory;
}

static void release_device_memory(void *context, void *memory) {
    (void)context;
    @autoreleasepool {
        [registered_buffers() removeObjectForKey:[NSValue valueWithPointer:memory]];
    }
}

static id<MTLBuffer> buffer_for(const float *memory) {
    return memory ? registered_buffers()[[NSValue valueWithPointer:memory]] : nil;
}

static void end_compute_encoding(MPSBackend *backend) {
    if (backend->computeEncoder) {
        [backend->computeEncoder endEncoding];
        backend->computeEncoder = nil;
    }
}

static id<MTLCommandBuffer> pending_command_buffer(MPSBackend *backend) {
    if (!backend->pendingCommandBuffer) {
        backend->pendingCommandBuffer = [backend->queue commandBuffer];
    }
    return backend->pendingCommandBuffer;
}

static id<MTLComputeCommandEncoder> compute_encoder(MPSBackend *backend) {
    if (!backend->computeEncoder) {
        backend->computeEncoder =
            [pending_command_buffer(backend) computeCommandEncoder];
    }
    return backend->computeEncoder;
}

static void synchronize_device(void *context) {
    MPSBackend *backend = (MPSBackend *)context;
    @autoreleasepool {
        end_compute_encoding(backend);
        id<MTLCommandBuffer> command_buffer = backend->pendingCommandBuffer;
        backend->pendingCommandBuffer = nil;
        if (!command_buffer) {
            return;
        }
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "Metal command buffer failed: %s\n",
                    command_buffer.error.localizedDescription.UTF8String);
            abort();
        }
    }
}

typedef struct {
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    uint32_t batch;
    uint32_t sequence_length;
    uint32_t d_model;
    uint32_t heads;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t patch_size;
    uint32_t parameter_index;
    uint32_t has_class_weights;
    float scale;
    float epsilon;
    float label_smoothing;
    float max_norm;
    float learning_rate;
    float beta1;
    float beta2;
    float weight_decay;
    float correction1;
    float correction2;
} KernelParameters;

static int fits_kernel_index(size_t value) {
    return value <= UINT32_MAX;
}

static int dispatch_kernel(MPSBackend *backend, NSString *name,
                           const id<MTLBuffer> *buffers, NSUInteger buffer_count,
                           const KernelParameters *parameters,
                           size_t thread_count) {
    for (NSUInteger i = 0; i < buffer_count; ++i) {
        if (!buffers[i]) {
            return -1;
        }
    }
    if (thread_count == 0 || !fits_kernel_index(thread_count)) {
        return -1;
    }
    id<MTLComputePipelineState> pipeline = backend->pipelines[name];
    id<MTLComputeCommandEncoder> encoder = compute_encoder(backend);
    if (!pipeline || !encoder) {
        return -1;
    }
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < buffer_count; ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [encoder setBytes:parameters length:sizeof(*parameters) atIndex:buffer_count];
    const NSUInteger threads_per_group =
        MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    return 0;
}

static int dispatch_threadgroups(MPSBackend *backend, NSString *name,
                                 const id<MTLBuffer> *buffers,
                                 NSUInteger buffer_count,
                                 const KernelParameters *parameters,
                                 MTLSize threadgroups,
                                 MTLSize threads_per_threadgroup,
                                 NSUInteger threadgroup_memory_bytes) {
    for (NSUInteger i = 0; i < buffer_count; ++i) {
        if (!buffers[i]) {
            return -1;
        }
    }
    id<MTLComputePipelineState> pipeline = backend->pipelines[name];
    const NSUInteger threads = threads_per_threadgroup.width *
                               threads_per_threadgroup.height *
                               threads_per_threadgroup.depth;
    if (!pipeline || threads > pipeline.maxTotalThreadsPerThreadgroup ||
        threadgroup_memory_bytes > backend->device.maxThreadgroupMemoryLength) {
        return -1;
    }
    id<MTLComputeCommandEncoder> encoder = compute_encoder(backend);
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < buffer_count; ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [encoder setBytes:parameters length:sizeof(*parameters) atIndex:buffer_count];
    if (threadgroup_memory_bytes > 0) {
        const NSUInteger alignment = 16;
        [encoder setThreadgroupMemoryLength:(threadgroup_memory_bytes +
                                             alignment - 1) /
                                            alignment * alignment
                                    atIndex:0];
    }
    [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threads_per_threadgroup];
    return 0;
}

static NSUInteger column_tile_rows(MPSBackend *backend, NSString *name) {
    const NSUInteger tile_columns = 32;
    const NSUInteger preferred_rows = 32;
    const NSUInteger available_rows =
        backend->pipelines[name].maxTotalThreadsPerThreadgroup / tile_columns;
    NSUInteger rows = 1;
    while (rows * 2 <= MIN(preferred_rows, available_rows)) {
        rows *= 2;
    }
    return rows;
}

static int dispatch_column_tiles(MPSBackend *backend, NSString *name,
                                 const id<MTLBuffer> *buffers,
                                 NSUInteger buffer_count,
                                 const KernelParameters *parameters,
                                 NSUInteger partial_sums_per_thread) {
    const NSUInteger tile_columns = 32;
    const NSUInteger tile_rows = column_tile_rows(backend, name);
    const NSUInteger tiles = (parameters->cols + tile_columns - 1) / tile_columns;
    return dispatch_threadgroups(
        backend, name, buffers, buffer_count, parameters,
        MTLSizeMake(tiles, 1, 1), MTLSizeMake(tile_columns, tile_rows, 1),
        partial_sums_per_thread * tile_columns * tile_rows * sizeof(float));
}

static int dispatch_simdgroup_rows(MPSBackend *backend, NSString *name,
                                   const id<MTLBuffer> *buffers,
                                   NSUInteger buffer_count,
                                   const KernelParameters *parameters) {
    const NSUInteger simd_width = 32;
    const NSUInteger rows_per_threadgroup = 8;
    const NSUInteger threadgroups =
        (parameters->rows + rows_per_threadgroup - 1) / rows_per_threadgroup;
    if (backend->pipelines[name].threadExecutionWidth != simd_width) {
        return -1;
    }
    return dispatch_threadgroups(
        backend, name, buffers, buffer_count, parameters,
        MTLSizeMake(threadgroups, 1, 1),
        MTLSizeMake(simd_width * rows_per_threadgroup, 1, 1), 0);
}

static int all_indices_fit(const Tensor *tensor) {
    return fits_kernel_index(tensor_numel(tensor));
}

static int device_accumulate_gradient(void *context, Tensor *destination,
                                      const Tensor *source) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(destination->grad),
                                         buffer_for(source->grad)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(destination);
        return all_indices_fit(destination)
            ? dispatch_kernel((MPSBackend *)context, @"accumulate_gradient",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static int device_residual(void *context, const Tensor *left,
                           const Tensor *right, Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(left->data),
                                         buffer_for(right->data),
                                         buffer_for(output->data)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(output);
        return all_indices_fit(output)
            ? dispatch_kernel((MPSBackend *)context, @"residual_forward",
                              buffers, 3, &parameters, parameters.count)
            : -1;
    }
}

static int device_residual_backward(void *context, Tensor *left, Tensor *right,
                                    const Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(left->grad),
                                         buffer_for(right->grad),
                                         buffer_for(output->grad)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(output);
        return all_indices_fit(output) && left->grad != right->grad
            ? dispatch_kernel((MPSBackend *)context, @"residual_backward",
                              buffers, 3, &parameters, parameters.count)
            : -1;
    }
}

static int device_bias_add(void *context, const Tensor *input,
                           const Tensor *bias, Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(input->data),
                                         buffer_for(bias->data),
                                         buffer_for(output->data)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(output);
        parameters.cols = (uint32_t)output->cols;
        return all_indices_fit(output)
            ? dispatch_kernel((MPSBackend *)context, @"bias_add_forward",
                              buffers, 3, &parameters, parameters.count)
            : -1;
    }
}

static int device_bias_add_backward(void *context, const Tensor *output,
                                    Tensor *bias_grad) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(output->grad),
                                         buffer_for(bias_grad->grad)};
        KernelParameters parameters = {};
        parameters.rows = (uint32_t)output->rows;
        parameters.cols = (uint32_t)output->cols;
        return all_indices_fit(output)
            ? dispatch_column_tiles((MPSBackend *)context, @"column_sums_tiled",
                                    buffers, 2, &parameters, 1)
            : -1;
    }
}

static int device_gelu(void *context, const Tensor *input, Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(input->data),
                                         buffer_for(output->data)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(output);
        return all_indices_fit(output)
            ? dispatch_kernel((MPSBackend *)context, @"gelu_forward", buffers,
                              2, &parameters, parameters.count)
            : -1;
    }
}

static int device_gelu_backward(void *context, const Tensor *input,
                                const Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(input->data),
                                         buffer_for(input->grad),
                                         buffer_for(output->grad)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(output);
        return all_indices_fit(output)
            ? dispatch_kernel((MPSBackend *)context, @"gelu_backward", buffers,
                              3, &parameters, parameters.count)
            : -1;
    }
}

static int device_layer_norm(void *context, const Tensor *input,
                             const Tensor *gamma, const Tensor *beta,
                             float epsilon, Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(input->data),
                                         buffer_for(gamma->data),
                                         buffer_for(beta->data),
                                         buffer_for(output->data)};
        KernelParameters parameters = {};
        parameters.rows = (uint32_t)input->rows;
        parameters.cols = (uint32_t)input->cols;
        parameters.epsilon = epsilon;
        return all_indices_fit(input)
            ? dispatch_simdgroup_rows((MPSBackend *)context,
                                      @"layer_norm_forward_simd", buffers, 4,
                                      &parameters)
            : -1;
    }
}

static id<MTLBuffer> statistics_buffer(MPSBackend *backend, size_t rows) {
    const NSUInteger required_bytes = (NSUInteger)(2 * rows * sizeof(float));
    if (!backend->statisticsBuffer ||
        backend->statisticsBuffer.length < required_bytes) {
        backend->statisticsBuffer =
            [backend->device newBufferWithLength:required_bytes
                                         options:MTLResourceStorageModePrivate];
    }
    return backend->statisticsBuffer;
}

static int device_layer_norm_backward(void *context, const Tensor *input,
                                      const Tensor *gamma, float epsilon,
                                      const Tensor *output, Tensor *input_grad,
                                      Tensor *gamma_grad, Tensor *beta_grad) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        if (!all_indices_fit(input)) {
            return -1;
        }
        id<MTLBuffer> statistics = statistics_buffer(backend, input->rows);
        const id<MTLBuffer> row_buffers[] = {
            buffer_for(input->data), buffer_for(gamma->data),
            buffer_for(output->grad), buffer_for(input_grad->grad), statistics,
        };
        const id<MTLBuffer> parameter_buffers[] = {
            buffer_for(input->data), buffer_for(output->grad), statistics,
            buffer_for(gamma_grad->grad), buffer_for(beta_grad->grad),
        };
        for (const id<MTLBuffer> &buffer : parameter_buffers) {
            if (!buffer) {
                return -1;
            }
        }
        KernelParameters parameters = {};
        parameters.rows = (uint32_t)input->rows;
        parameters.cols = (uint32_t)input->cols;
        parameters.epsilon = epsilon;
        for (const id<MTLBuffer> &buffer : row_buffers) {
            if (!buffer) {
                return -1;
            }
        }
        const NSUInteger gamma_and_beta_sums = 2;
        return dispatch_simdgroup_rows(backend, @"layer_norm_backward_rows_simd",
                                       row_buffers, 5, &parameters) ||
            dispatch_column_tiles(backend,
                                  @"layer_norm_parameter_gradients_tiled",
                                  parameter_buffers, 5, &parameters,
                                  gamma_and_beta_sums)
            ? -1
            : 0;
    }
}

static KernelParameters head_parameters(size_t batch, size_t sequence_length,
                                        size_t d_model, size_t heads) {
    KernelParameters parameters = {};
    parameters.count = (uint32_t)(batch * sequence_length * d_model);
    parameters.batch = (uint32_t)batch;
    parameters.sequence_length = (uint32_t)sequence_length;
    parameters.d_model = (uint32_t)d_model;
    parameters.heads = (uint32_t)heads;
    return parameters;
}

static int device_split_heads(void *context, const Tensor *projected,
                              Tensor *query, Tensor *key, Tensor *value,
                              size_t batch, size_t sequence_length,
                              size_t d_model, size_t heads) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {
            buffer_for(projected->data), buffer_for(query->data),
            buffer_for(key->data), buffer_for(value->data),
        };
        const KernelParameters parameters =
            head_parameters(batch, sequence_length, d_model, heads);
        return all_indices_fit(projected)
            ? dispatch_kernel((MPSBackend *)context, @"split_heads_forward",
                              buffers, 4, &parameters, parameters.count)
            : -1;
    }
}

static int device_split_heads_backward(void *context, Tensor *projected,
                                       const Tensor *query, const Tensor *key,
                                       const Tensor *value, size_t batch,
                                       size_t sequence_length, size_t d_model,
                                       size_t heads) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {
            buffer_for(projected->grad), buffer_for(query->grad),
            buffer_for(key->grad), buffer_for(value->grad),
        };
        const KernelParameters parameters =
            head_parameters(batch, sequence_length, d_model, heads);
        return all_indices_fit(projected)
            ? dispatch_kernel((MPSBackend *)context, @"split_heads_backward",
                              buffers, 4, &parameters, parameters.count)
            : -1;
    }
}

static int device_merge_heads(void *context, const Tensor *attended,
                              Tensor *merged, size_t batch,
                              size_t sequence_length, size_t d_model,
                              size_t heads) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(attended->data),
                                         buffer_for(merged->data)};
        const KernelParameters parameters =
            head_parameters(batch, sequence_length, d_model, heads);
        return all_indices_fit(merged)
            ? dispatch_kernel((MPSBackend *)context, @"merge_heads_forward",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static int device_merge_heads_backward(void *context, Tensor *attended,
                                       const Tensor *merged, size_t batch,
                                       size_t sequence_length, size_t d_model,
                                       size_t heads) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(attended->grad),
                                         buffer_for(merged->grad)};
        const KernelParameters parameters =
            head_parameters(batch, sequence_length, d_model, heads);
        return all_indices_fit(merged)
            ? dispatch_kernel((MPSBackend *)context, @"merge_heads_backward",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static KernelParameters attention_parameters(const Tensor *query,
                                             size_t sequence_length,
                                             float scale) {
    KernelParameters parameters = {};
    parameters.rows = (uint32_t)query->rows;
    parameters.cols = (uint32_t)query->cols;
    parameters.sequence_length = (uint32_t)sequence_length;
    parameters.scale = scale;
    return parameters;
}

static const NSUInteger kAttentionThreadsPerGroup = 256;

static NSUInteger attention_forward_shared_bytes(size_t sequence_length,
                                                 size_t head_dimension) {
    return (3 * sequence_length * head_dimension +
            sequence_length * sequence_length) * sizeof(float);
}

static NSUInteger attention_backward_shared_bytes(size_t sequence_length,
                                                  size_t head_dimension) {
    return (4 * sequence_length * head_dimension +
            sequence_length * sequence_length + sequence_length) *
           sizeof(float);
}

static int device_multi_head_attention(void *context, const Tensor *query,
                                       const Tensor *key, const Tensor *value,
                                       size_t batch, size_t heads,
                                       size_t sequence_length, float scale,
                                       Tensor *probabilities, Tensor *output) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {
            buffer_for(query->data), buffer_for(key->data),
            buffer_for(value->data), buffer_for(probabilities->data),
            buffer_for(output->data),
        };
        MPSBackend *backend = (MPSBackend *)context;
        const KernelParameters parameters =
            attention_parameters(query, sequence_length, scale);
        if (!all_indices_fit(probabilities)) {
            return -1;
        }
        const NSUInteger shared_bytes =
            attention_forward_shared_bytes(sequence_length, query->cols);
        if (shared_bytes <= backend->device.maxThreadgroupMemoryLength) {
            return dispatch_threadgroups(
                backend, @"attention_forward_shared", buffers, 5, &parameters,
                MTLSizeMake(batch * heads, 1, 1),
                MTLSizeMake(kAttentionThreadsPerGroup, 1, 1),
                shared_bytes);
        }
        if (query->cols <= 64) {
            const NSUInteger blocks =
                (sequence_length + 31) / 32;
            return dispatch_threadgroups(
                backend, @"attention_forward_flash", buffers, 5, &parameters,
                MTLSizeMake(blocks, batch * heads, 1),
                MTLSizeMake(kAttentionThreadsPerGroup, 1, 1), 0);
        }
        return dispatch_kernel(backend, @"attention_forward", buffers, 5,
                               &parameters, parameters.rows);
    }
}

static int device_multi_head_attention_backward(
    void *context, const Tensor *query, const Tensor *key, const Tensor *value,
    size_t batch, size_t heads, size_t sequence_length, float scale,
    const Tensor *probabilities, const Tensor *output, Tensor *query_grad,
    Tensor *key_grad, Tensor *value_grad) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        const NSUInteger shared_bytes =
            attention_backward_shared_bytes(sequence_length, query->cols);
        if (shared_bytes <= backend->device.maxThreadgroupMemoryLength) {
            const id<MTLBuffer> fused_buffers[] = {
                buffer_for(query->data), buffer_for(key->data),
                buffer_for(value->data), buffer_for(probabilities->data),
                buffer_for(output->data), buffer_for(output->grad),
                buffer_for(query_grad->grad), buffer_for(key_grad->grad),
                buffer_for(value_grad->grad),
            };
            const KernelParameters parameters =
                attention_parameters(query, sequence_length, scale);
            return all_indices_fit(probabilities)
                ? dispatch_threadgroups(
                      backend, @"attention_backward_shared", fused_buffers, 9,
                      &parameters, MTLSizeMake(batch * heads, 1, 1),
                      MTLSizeMake(kAttentionThreadsPerGroup, 1, 1),
                      shared_bytes)
                : -1;
        }
        if (query->cols <= 64) {
            const id<MTLBuffer> query_buffers[] = {
                buffer_for(key->data), buffer_for(value->data),
                buffer_for(probabilities->data), buffer_for(output->data),
                buffer_for(output->grad), buffer_for(probabilities->grad),
                buffer_for(query_grad->grad),
            };
            const id<MTLBuffer> key_buffers[] = {
                buffer_for(query->data), buffer_for(probabilities->data),
                buffer_for(output->grad), buffer_for(probabilities->grad),
                buffer_for(key_grad->grad), buffer_for(value_grad->grad),
            };
            for (const id<MTLBuffer> &buffer : query_buffers) {
                if (!buffer) {
                    return -1;
                }
            }
            for (const id<MTLBuffer> &buffer : key_buffers) {
                if (!buffer) {
                    return -1;
                }
            }
            const KernelParameters parameters =
                attention_parameters(query, sequence_length, scale);
            const MTLSize grid = MTLSizeMake(
                (sequence_length + 31) / 32, batch * heads, 1);
            return all_indices_fit(probabilities) &&
                           dispatch_threadgroups(
                               backend, @"attention_backward_flash_queries",
                               query_buffers, 7, &parameters, grid,
                               MTLSizeMake(kAttentionThreadsPerGroup, 1, 1),
                               0) == 0 &&
                           dispatch_threadgroups(
                               backend, @"attention_backward_flash_keys",
                               key_buffers, 6, &parameters, grid,
                               MTLSizeMake(kAttentionThreadsPerGroup, 1, 1),
                               0) == 0
                ? 0
                : -1;
        }
        id<MTLBuffer> score_gradients = buffer_for(probabilities->grad);
        const id<MTLBuffer> query_buffers[] = {
            buffer_for(key->data), buffer_for(value->data),
            buffer_for(probabilities->data), buffer_for(output->grad),
            score_gradients, buffer_for(query_grad->grad),
        };
        const id<MTLBuffer> key_value_buffers[] = {
            buffer_for(query->data), buffer_for(probabilities->data),
            buffer_for(output->grad), score_gradients,
            buffer_for(key_grad->grad), buffer_for(value_grad->grad),
        };
        for (const id<MTLBuffer> &buffer : key_value_buffers) {
            if (!buffer) {
                return -1;
            }
        }
        const KernelParameters parameters =
            attention_parameters(query, sequence_length, scale);
        return all_indices_fit(probabilities) &&
                       dispatch_kernel(backend, @"attention_backward_query",
                                       query_buffers, 6, &parameters,
                                       parameters.rows) == 0 &&
                       dispatch_kernel(backend, @"attention_backward_key_value",
                                       key_value_buffers, 6, &parameters,
                                       parameters.rows) == 0
            ? 0
            : -1;
    }
}

static MPSMatrix *tensor_matrix(id<MTLBuffer> buffer, const Tensor *tensor) {
    if (!buffer) {
        return nil;
    }
    MPSMatrixDescriptor *descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:tensor->rows
                                               columns:tensor->cols
                                              rowBytes:tensor->cols * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}

static MPSMatrixMultiplication *tensor_product(MPSBackend *backend,
                                               BOOL transpose_left,
                                               BOOL transpose_right,
                                               size_t result_rows,
                                               size_t result_cols,
                                               size_t interior_cols,
                                               double beta) {
    NSString *key = [NSString stringWithFormat:@"%d-%d-%zu-%zu-%zu-%g",
                                               transpose_left, transpose_right,
                                               result_rows, result_cols,
                                               interior_cols, beta];
    MPSMatrixMultiplication *product = backend->tensorProducts[key];
    if (!product) {
        product = create_product(backend->device, transpose_left,
                                 transpose_right, result_rows, result_cols,
                                 interior_cols, beta);
        if (product) {
            backend->tensorProducts[key] = product;
        }
    }
    return product;
}

static int device_gemm(void *context, const Tensor *left, const Tensor *right,
                       Tensor *output) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        MPSMatrix *left_matrix = tensor_matrix(buffer_for(left->data), left);
        MPSMatrix *right_matrix = tensor_matrix(buffer_for(right->data), right);
        MPSMatrix *output_matrix =
            tensor_matrix(buffer_for(output->data), output);
        MPSMatrixMultiplication *product = tensor_product(
            backend, NO, NO, left->rows, right->cols, left->cols, 0.0);
        if (!left_matrix || !right_matrix || !output_matrix || !product) {
            return -1;
        }
        end_compute_encoding(backend);
        [product encodeToCommandBuffer:pending_command_buffer(backend)
                            leftMatrix:left_matrix
                           rightMatrix:right_matrix
                          resultMatrix:output_matrix];
        return 0;
    }
}

static int device_gemm_backward(void *context, Tensor *left, Tensor *right,
                                const Tensor *output) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        const double accumulate_into_result = 1.0;
        MPSMatrix *left_matrix = tensor_matrix(buffer_for(left->data), left);
        MPSMatrix *right_matrix = tensor_matrix(buffer_for(right->data), right);
        MPSMatrix *output_gradient =
            tensor_matrix(buffer_for(output->grad), output);
        MPSMatrix *left_gradient = tensor_matrix(buffer_for(left->grad), left);
        MPSMatrix *right_gradient =
            tensor_matrix(buffer_for(right->grad), right);
        MPSMatrixMultiplication *left_product =
            tensor_product(backend, NO, YES, left->rows, left->cols,
                           right->cols, accumulate_into_result);
        MPSMatrixMultiplication *right_product =
            tensor_product(backend, YES, NO, right->rows, right->cols,
                           left->rows, accumulate_into_result);
        if (!left_matrix || !right_matrix || !output_gradient ||
            !left_gradient || !right_gradient || !left_product ||
            !right_product) {
            return -1;
        }
        end_compute_encoding(backend);
        id<MTLCommandBuffer> command_buffer = pending_command_buffer(backend);
        [left_product encodeToCommandBuffer:command_buffer
                                 leftMatrix:output_gradient
                                rightMatrix:right_matrix
                               resultMatrix:left_gradient];
        [right_product encodeToCommandBuffer:command_buffer
                                  leftMatrix:left_matrix
                                 rightMatrix:output_gradient
                                resultMatrix:right_gradient];
        return 0;
    }
}

static id<MTLBuffer> scratch_buffer(MPSBackend *backend, NSString *name,
                                    size_t float_count) {
    const NSUInteger required_bytes =
        (NSUInteger)(MAX(float_count, (size_t)1) * sizeof(float));
    id<MTLBuffer> buffer = backend->scratchBuffers[name];
    if (!buffer || buffer.length < required_bytes) {
        buffer = [backend->device newBufferWithLength:required_bytes
                                              options:MTLResourceStorageModePrivate];
        if (buffer) {
            backend->scratchBuffers[name] = buffer;
        }
    }
    return buffer;
}

static int device_zero_gradient(void *context, Tensor *tensor) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(tensor->grad)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(tensor);
        return all_indices_fit(tensor)
            ? dispatch_kernel((MPSBackend *)context, @"zero_gradient", buffers,
                              1, &parameters, parameters.count)
            : -1;
    }
}

static int device_extract_patches(void *context, const Tensor *images,
                                  Tensor *patches, size_t channels,
                                  size_t height, size_t width,
                                  size_t patch_size) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(images->data),
                                         buffer_for(patches->data)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(patches);
        parameters.cols = (uint32_t)patches->cols;
        parameters.channels = (uint32_t)channels;
        parameters.height = (uint32_t)height;
        parameters.width = (uint32_t)width;
        parameters.patch_size = (uint32_t)patch_size;
        return all_indices_fit(patches) && all_indices_fit(images)
            ? dispatch_kernel((MPSBackend *)context, @"extract_patches",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static KernelParameters token_parameters(const Tensor *tokens,
                                         const Tensor *positional,
                                         size_t batch) {
    KernelParameters parameters = {};
    parameters.count = (uint32_t)tensor_numel(tokens);
    parameters.batch = (uint32_t)batch;
    parameters.sequence_length = (uint32_t)positional->rows;
    parameters.d_model = (uint32_t)tokens->cols;
    return parameters;
}

static int device_token_embedding(void *context, const Tensor *patch_tokens,
                                  const Tensor *cls_token,
                                  const Tensor *positional, Tensor *tokens,
                                  size_t batch) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {
            buffer_for(patch_tokens->data), buffer_for(cls_token->data),
            buffer_for(positional->data), buffer_for(tokens->data),
        };
        const KernelParameters parameters =
            token_parameters(tokens, positional, batch);
        return all_indices_fit(tokens)
            ? dispatch_kernel((MPSBackend *)context, @"token_embedding_forward",
                              buffers, 4, &parameters, parameters.count)
            : -1;
    }
}

static int device_token_embedding_backward(void *context, Tensor *patch_tokens,
                                           Tensor *cls_token,
                                           Tensor *positional,
                                           const Tensor *tokens, size_t batch) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        const id<MTLBuffer> patch_buffers[] = {buffer_for(patch_tokens->grad),
                                               buffer_for(tokens->grad)};
        const id<MTLBuffer> parameter_buffers[] = {
            buffer_for(cls_token->grad), buffer_for(positional->grad),
            buffer_for(tokens->grad),
        };
        for (const id<MTLBuffer> &buffer : parameter_buffers) {
            if (!buffer || !patch_buffers[0]) {
                return -1;
            }
        }
        KernelParameters patch_parameters =
            token_parameters(tokens, positional, batch);
        patch_parameters.count = (uint32_t)tensor_numel(patch_tokens);
        KernelParameters parameter_parameters =
            token_parameters(tokens, positional, batch);
        parameter_parameters.count = (uint32_t)tensor_numel(positional);
        return all_indices_fit(tokens) &&
                       dispatch_kernel(backend,
                                       @"token_embedding_backward_patches",
                                       patch_buffers, 2, &patch_parameters,
                                       patch_parameters.count) == 0 &&
                       dispatch_kernel(backend,
                                       @"token_embedding_backward_parameters",
                                       parameter_buffers, 3,
                                       &parameter_parameters,
                                       parameter_parameters.count) == 0
            ? 0
            : -1;
    }
}

static int device_gather_rows(void *context, const Tensor *source,
                              Tensor *destination, size_t row_stride) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(source->data),
                                         buffer_for(destination->data)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(destination);
        parameters.cols = (uint32_t)destination->cols;
        parameters.sequence_length = (uint32_t)row_stride;
        return all_indices_fit(source)
            ? dispatch_kernel((MPSBackend *)context, @"gather_rows_forward",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static int device_gather_rows_backward(void *context, Tensor *source,
                                       const Tensor *destination,
                                       size_t row_stride) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {buffer_for(source->grad),
                                         buffer_for(destination->grad)};
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(destination);
        parameters.cols = (uint32_t)destination->cols;
        parameters.sequence_length = (uint32_t)row_stride;
        return all_indices_fit(source)
            ? dispatch_kernel((MPSBackend *)context, @"gather_rows_backward",
                              buffers, 2, &parameters, parameters.count)
            : -1;
    }
}

static int device_softmax_cross_entropy(void *context, const Tensor *logits,
                                        const Tensor *targets,
                                        const Tensor *class_weights,
                                        float label_smoothing, Tensor *loss,
                                        Tensor *logits_grad) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        id<MTLBuffer> logits_buffer = buffer_for(logits->data);
        id<MTLBuffer> row_losses =
            scratch_buffer(backend, @"row_losses", logits->rows);
        const id<MTLBuffer> row_buffers[] = {
            logits_buffer, buffer_for(targets->data),
            class_weights ? buffer_for(class_weights->data) : logits_buffer,
            buffer_for(logits_grad->grad), row_losses,
        };
        const id<MTLBuffer> loss_buffers[] = {row_losses,
                                              buffer_for(loss->data)};
        for (const id<MTLBuffer> &buffer : row_buffers) {
            if (!buffer || !loss_buffers[1]) {
                return -1;
            }
        }
        KernelParameters parameters = {};
        parameters.rows = (uint32_t)logits->rows;
        parameters.cols = (uint32_t)logits->cols;
        parameters.has_class_weights = class_weights ? 1u : 0u;
        parameters.label_smoothing = label_smoothing;
        return all_indices_fit(logits) &&
                       dispatch_kernel(backend, @"softmax_cross_entropy_rows",
                                       row_buffers, 5, &parameters,
                                       parameters.rows) == 0 &&
                       dispatch_kernel(backend, @"mean_loss", loss_buffers, 2,
                                       &parameters, 1) == 0
            ? 0
            : -1;
    }
}

static int device_adamw_step(void *context, Parameter *parameter,
                             float learning_rate, float beta1, float beta2,
                             float epsilon, float weight_decay,
                             float correction1, float correction2) {
    @autoreleasepool {
        const id<MTLBuffer> buffers[] = {
            buffer_for(parameter->value.data), buffer_for(parameter->value.grad),
            buffer_for(parameter->m), buffer_for(parameter->v),
        };
        KernelParameters parameters = {};
        parameters.count = (uint32_t)tensor_numel(&parameter->value);
        parameters.learning_rate = learning_rate;
        parameters.beta1 = beta1;
        parameters.beta2 = beta2;
        parameters.epsilon = epsilon;
        parameters.weight_decay = weight_decay;
        parameters.correction1 = correction1;
        parameters.correction2 = correction2;
        return all_indices_fit(&parameter->value)
            ? dispatch_kernel((MPSBackend *)context, @"adamw_step", buffers, 4,
                              &parameters, parameters.count)
            : -1;
    }
}

typedef struct {
    uint64_t value;
    uint64_t gradient;
    uint64_t first_moment;
    uint64_t second_moment;
    uint32_t count;
    uint32_t element_offset;
    float weight_decay;
    float correction1;
    float correction2;
    uint32_t padding;
} ParameterView;

static_assert(sizeof(ParameterView) == 56, "ParameterView must match Metal");

typedef struct {
    id<MTLBuffer> table;
    NSMutableArray<id<MTLBuffer>> *resources;
    size_t view_count;
    size_t element_count;
} ParameterViewTable;

static uint64_t gpu_address_of(id<MTLBuffer> buffer) {
    if (@available(macOS 13.0, *)) {
        return buffer ? buffer.gpuAddress : 0;
    }
    return 0;
}

static int append_view_buffer(ParameterViewTable *table, const float *memory,
                              uint64_t *address) {
    if (!memory) {
        *address = 0;
        return 0;
    }
    id<MTLBuffer> buffer = buffer_for(memory);
    *address = gpu_address_of(buffer);
    if (*address == 0) {
        return -1;
    }
    [table->resources addObject:buffer];
    return 0;
}

static int build_view_table(MPSBackend *backend, const Tensor *const *tensors,
                            const float *const *first_moments,
                            const float *const *second_moments,
                            const float *weight_decays,
                            const float *corrections1,
                            const float *corrections2, size_t view_count,
                            ParameterViewTable *table) {
    if (view_count == 0 || !fits_kernel_index(view_count)) {
        return -1;
    }
    ParameterView *views =
        (ParameterView *)calloc(view_count, sizeof(ParameterView));
    if (!views) {
        return -1;
    }
    table->resources = [[NSMutableArray alloc] init];
    table->view_count = view_count;
    size_t element_offset = 0;
    int result = 0;
    for (size_t i = 0; result == 0 && i < view_count; ++i) {
        const size_t count = tensor_numel(tensors[i]);
        ParameterView *view = &views[i];
        result = !fits_kernel_index(element_offset + count) ||
            append_view_buffer(table, tensors[i]->data, &view->value) ||
            append_view_buffer(table, tensors[i]->grad, &view->gradient) ||
            append_view_buffer(table, first_moments ? first_moments[i] : NULL,
                               &view->first_moment) ||
            append_view_buffer(table, second_moments ? second_moments[i] : NULL,
                               &view->second_moment);
        view->count = (uint32_t)count;
        view->element_offset = (uint32_t)element_offset;
        view->weight_decay = weight_decays ? weight_decays[i] : 0.0f;
        view->correction1 = corrections1 ? corrections1[i] : 1.0f;
        view->correction2 = corrections2 ? corrections2[i] : 1.0f;
        element_offset += count;
    }
    table->element_count = element_offset;
    if (result == 0) {
        table->table =
            [backend->device newBufferWithBytes:views
                                         length:view_count * sizeof(ParameterView)
                                        options:MTLResourceStorageModeShared];
        result = table->table && element_offset > 0 ? 0 : -1;
    }
    free(views);
    return result;
}

static int dispatch_over_views(MPSBackend *backend, NSString *name,
                               const ParameterViewTable *table,
                               id<MTLBuffer> extra_buffer,
                               KernelParameters *parameters,
                               MTLSize threadgroups,
                               MTLSize threads_per_threadgroup,
                               BOOL one_thread_per_element) {
    id<MTLComputePipelineState> pipeline = backend->pipelines[name];
    id<MTLComputeCommandEncoder> encoder = compute_encoder(backend);
    if (!pipeline || !encoder) {
        return -1;
    }
    parameters->parameter_index = (uint32_t)table->view_count;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:table->table offset:0 atIndex:0];
    NSUInteger parameters_index = 1;
    if (extra_buffer) {
        [encoder setBuffer:extra_buffer offset:0 atIndex:1];
        parameters_index = 2;
    }
    [encoder setBytes:parameters length:sizeof(*parameters)
              atIndex:parameters_index];
    for (id<MTLBuffer> buffer in table->resources) {
        [encoder useResource:buffer
                       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    if (one_thread_per_element) {
        const NSUInteger threads =
            MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        [encoder dispatchThreads:MTLSizeMake(table->element_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        [encoder dispatchThreadgroups:threadgroups
                threadsPerThreadgroup:threads_per_threadgroup];
    }
    return 0;
}

static int dispatch_elementwise_views(MPSBackend *backend, NSString *name,
                                      const ParameterViewTable *table,
                                      id<MTLBuffer> extra_buffer,
                                      KernelParameters *parameters) {
    parameters->count = (uint32_t)table->element_count;
    return dispatch_over_views(backend, name, table, extra_buffer, parameters,
                               MTLSizeMake(0, 0, 0), MTLSizeMake(0, 0, 0), YES);
}

static id<MTLBuffer> encode_gradient_clip_scale(MPSBackend *backend,
                                                const ParameterViewTable *table,
                                                float max_norm) {
    const NSUInteger reduction_threads = 256;
    id<MTLBuffer> partials =
        scratch_buffer(backend, @"gradient_norm_partials", table->view_count);
    id<MTLBuffer> scale = scratch_buffer(backend, @"gradient_scale", 1);
    if (!partials || !scale ||
        backend->pipelines[@"squared_norm_views"].maxTotalThreadsPerThreadgroup <
            reduction_threads) {
        return nil;
    }
    KernelParameters norm_parameters = {};
    KernelParameters scale_parameters = {};
    scale_parameters.count = (uint32_t)table->view_count;
    scale_parameters.max_norm = max_norm;
    const id<MTLBuffer> scale_buffers[] = {partials, scale};
    return dispatch_over_views(backend, @"squared_norm_views", table, partials,
                               &norm_parameters,
                               MTLSizeMake(table->view_count, 1, 1),
                               MTLSizeMake(reduction_threads, 1, 1), NO) == 0 &&
                   dispatch_kernel(backend, @"clip_scale", scale_buffers, 2,
                                   &scale_parameters, 1) == 0
        ? scale
        : nil;
}

static const Tensor **parameter_values(Parameter *const *parameters,
                                       size_t parameter_count) {
    const Tensor **values =
        (const Tensor **)malloc(parameter_count * sizeof(*values));
    for (size_t i = 0; values && i < parameter_count; ++i) {
        values[i] = &parameters[i]->value;
    }
    return values;
}

static int device_zero_gradients(void *context, Tensor *const *tensors,
                                 size_t tensor_count) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        ParameterViewTable table = {};
        if (build_view_table(backend, (const Tensor *const *)tensors, NULL, NULL,
                             NULL, NULL, NULL, tensor_count, &table) != 0) {
            return -1;
        }
        KernelParameters parameters = {};
        return dispatch_elementwise_views(backend, @"zero_gradients_views",
                                          &table, nil, &parameters);
    }
}

static int device_clip_gradient_norm_views(void *context,
                                           Parameter *const *parameters,
                                           size_t parameter_count,
                                           float max_norm) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        const Tensor **values = parameter_values(parameters, parameter_count);
        ParameterViewTable table = {};
        const int built = values
            ? build_view_table(backend, values, NULL, NULL, NULL, NULL, NULL,
                               parameter_count, &table)
            : -1;
        free(values);
        if (built != 0) {
            return -1;
        }
        id<MTLBuffer> scale = encode_gradient_clip_scale(backend, &table, max_norm);
        KernelParameters parameters = {};
        return scale ? dispatch_elementwise_views(backend,
                                                  @"scale_gradients_views",
                                                  &table, scale, &parameters)
                     : -1;
    }
}

static int device_adamw_step_parameters(void *context,
                                        Parameter *const *parameters,
                                        const float *weight_decays,
                                        size_t parameter_count,
                                        float learning_rate, float beta1,
                                        float beta2, float epsilon,
                                        float max_gradient_norm) {
    @autoreleasepool {
        MPSBackend *backend = (MPSBackend *)context;
        const Tensor **values = parameter_values(parameters, parameter_count);
        const float **first_moments =
            (const float **)malloc(parameter_count * sizeof(float *));
        const float **second_moments =
            (const float **)malloc(parameter_count * sizeof(float *));
        float *corrections = (float *)malloc(2 * parameter_count * sizeof(float));
        int built = -1;
        ParameterViewTable table = {};
        if (values && first_moments && second_moments && corrections) {
            for (size_t i = 0; i < parameter_count; ++i) {
                first_moments[i] = parameters[i]->m;
                second_moments[i] = parameters[i]->v;
                corrections[i] =
                    1.0f - powf(beta1, (float)parameters[i]->step);
                corrections[parameter_count + i] =
                    1.0f - powf(beta2, (float)parameters[i]->step);
            }
            built = build_view_table(backend, values, first_moments,
                                     second_moments, weight_decays, corrections,
                                     corrections + parameter_count,
                                     parameter_count, &table);
        }
        free(values);
        free(first_moments);
        free(second_moments);
        free(corrections);
        if (built != 0) {
            return -1;
        }
        id<MTLBuffer> scale = max_gradient_norm > 0.0f
            ? encode_gradient_clip_scale(backend, &table, max_gradient_norm)
            : table.table;
        if (!scale) {
            return -1;
        }
        KernelParameters adam_parameters = {};
        adam_parameters.learning_rate = learning_rate;
        adam_parameters.beta1 = beta1;
        adam_parameters.beta2 = beta2;
        adam_parameters.epsilon = epsilon;
        adam_parameters.max_norm = max_gradient_norm;
        return dispatch_elementwise_views(backend, @"adamw_views", &table,
                                          scale, &adam_parameters);
    }
}

static int compile_pipelines(MPSBackend *backend) {
    NSString *source =
        [[NSString alloc] initWithBytes:metal_kernels_metal
                                 length:metal_kernels_metal_len
                               encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 13.0, *)) {
        options.languageVersion = MTLLanguageVersion3_0;
    }
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [backend->device newLibraryWithSource:source
                                                          options:options
                                                            error:&error];
    if (!library) {
        fprintf(stderr, "Metal kernel compilation failed: %s\n",
                error.localizedDescription.UTF8String);
        return -1;
    }
    backend->pipelines = [[NSMutableDictionary alloc] init];
    for (NSString *name in library.functionNames) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        id<MTLComputePipelineState> pipeline =
            [backend->device newComputePipelineStateWithFunction:function
                                                           error:&error];
        if (!pipeline) {
            fprintf(stderr, "Metal pipeline %s failed: %s\n", name.UTF8String,
                    error.localizedDescription.UTF8String);
            return -1;
        }
        backend->pipelines[name] = pipeline;
    }
    return 0;
}

int mps_backend_enable_device_execution(MPSBackend *backend) {
    @autoreleasepool {
        if (!mps_backend_available(backend) ||
            (!backend->pipelines && compile_pipelines(backend) != 0)) {
            return -1;
        }
        backend->tensorProducts = [[NSMutableDictionary alloc] init];
        backend->scratchBuffers = [[NSMutableDictionary alloc] init];
        OpsDeviceBackend device_backend = {};
        device_backend.context = backend;
        device_backend.gemm = device_gemm;
        device_backend.gemm_backward = device_gemm_backward;
        device_backend.residual = device_residual;
        device_backend.residual_backward = device_residual_backward;
        device_backend.accumulate_gradient = device_accumulate_gradient;
        device_backend.bias_add = device_bias_add;
        device_backend.bias_add_backward = device_bias_add_backward;
        device_backend.layer_norm = device_layer_norm;
        device_backend.layer_norm_backward = device_layer_norm_backward;
        device_backend.gelu = device_gelu;
        device_backend.gelu_backward = device_gelu_backward;
        device_backend.split_heads = device_split_heads;
        device_backend.split_heads_backward = device_split_heads_backward;
        device_backend.merge_heads = device_merge_heads;
        device_backend.merge_heads_backward = device_merge_heads_backward;
        device_backend.multi_head_attention = device_multi_head_attention;
        device_backend.multi_head_attention_backward =
            device_multi_head_attention_backward;
        device_backend.zero_gradient = device_zero_gradient;
        device_backend.extract_patches = device_extract_patches;
        device_backend.token_embedding = device_token_embedding;
        device_backend.token_embedding_backward =
            device_token_embedding_backward;
        device_backend.gather_rows = device_gather_rows;
        device_backend.gather_rows_backward = device_gather_rows_backward;
        device_backend.softmax_cross_entropy = device_softmax_cross_entropy;
        device_backend.clip_gradient_norm = device_clip_gradient_norm_views;
        device_backend.adamw_step = device_adamw_step;
        device_backend.zero_gradients = device_zero_gradients;
        device_backend.adamw_step_parameters = device_adamw_step_parameters;
        device_backend.synchronize = synchronize_device;
        backend->deviceBackend = device_backend;
        ops_set_device_backend(&backend->deviceBackend);
        tensor_set_allocator(allocate_device_memory, release_device_memory,
                             backend);
        return 0;
    }
}

void mps_backend_disable_device_execution(MPSBackend *backend) {
    if (!backend || !backend->deviceBackend.context) {
        return;
    }
    ops_synchronize();
    ops_set_device_backend(NULL);
    tensor_set_allocator(NULL, NULL, NULL);
    backend->deviceBackend = OpsDeviceBackend{};
}

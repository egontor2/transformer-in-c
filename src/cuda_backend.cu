#include "cuda_backend.h"

extern "C" {
#include "ops.h"
#include "tensor.h"
}

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <unordered_set>
#include <vector>

namespace {

struct KernelParameters {
    unsigned count;
    unsigned rows;
    unsigned cols;
    unsigned batch;
    unsigned sequence_length;
    unsigned d_model;
    unsigned heads;
    unsigned channels;
    unsigned height;
    unsigned width;
    unsigned patch_size;
    unsigned parameter_index;
    unsigned has_class_weights;
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
};

struct ParameterView {
    float *value;
    float *gradient;
    float *first_moment;
    float *second_moment;
    unsigned count;
    unsigned element_offset;
    float weight_decay;
    float correction1;
    float correction2;
};

constexpr unsigned kThreadsPerBlock = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr unsigned kRowsPerLayerNormBlock = kThreadsPerBlock / kWarpSize;
constexpr unsigned kTileColumns = 32;
constexpr unsigned kTileRows = 32;
constexpr unsigned kFlashBlockItems = 32;
constexpr unsigned kFlashLanesPerItem = 8;
constexpr unsigned kFlashMaxHeadDimension = 64;
constexpr float kGeluCubicCoefficient = 0.044715f;
constexpr float kGeluTanhScale = 0.7978845608028654f;

__device__ unsigned global_thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float warp_sum(float value) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ float warp_max(float value) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, offset));
    }
    return value;
}

__global__ void accumulate_gradient(float *destination, const float *source,
                                    KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        destination[i] += source[i];
    }
}

__global__ void residual_forward(const float *left, const float *right,
                                 float *output, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        output[i] = left[i] + right[i];
    }
}

__global__ void residual_backward(float *left_grad, float *right_grad,
                                  const float *output_grad,
                                  KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        left_grad[i] += output_grad[i];
        right_grad[i] += output_grad[i];
    }
}

__global__ void bias_add_forward(const float *input, const float *bias,
                                 float *output, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        output[i] = input[i] + bias[i % p.cols];
    }
}

__global__ void gelu_forward(const float *input, float *output,
                             KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        const float x = input[i];
        output[i] = 0.5f * x *
            (1.0f + tanhf(kGeluTanhScale *
                          (x + kGeluCubicCoefficient * x * x * x)));
    }
}

__global__ void gelu_backward(const float *input, float *input_grad,
                              const float *output_grad, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const float x = input[i];
    const float tanh_inner = tanhf(
        kGeluTanhScale * (x + kGeluCubicCoefficient * x * x * x));
    const float derivative = 0.5f * (1.0f + tanh_inner) +
        0.5f * x * (1.0f - tanh_inner * tanh_inner) * kGeluTanhScale *
        (1.0f + 3.0f * kGeluCubicCoefficient * x * x);
    input_grad[i] += output_grad[i] * derivative;
}

__device__ unsigned head_major_index(unsigned row, unsigned feature,
                                     const KernelParameters &p) {
    const unsigned head_dimension = p.d_model / p.heads;
    const unsigned sample = row / p.sequence_length;
    const unsigned position = row % p.sequence_length;
    const unsigned head = feature / head_dimension;
    const unsigned dimension = feature % head_dimension;
    return ((sample * p.heads + head) * p.sequence_length + position) *
           head_dimension + dimension;
}

__global__ void split_heads_forward(const float *projected, float *query,
                                    float *key, float *value,
                                    KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned row = i / p.d_model;
    const unsigned feature = i % p.d_model;
    const unsigned target = head_major_index(row, feature, p);
    const float *source = projected + row * 3 * p.d_model;
    query[target] = source[feature];
    key[target] = source[p.d_model + feature];
    value[target] = source[2 * p.d_model + feature];
}

__global__ void split_heads_backward(float *projected_grad,
                                     const float *query_grad,
                                     const float *key_grad,
                                     const float *value_grad,
                                     KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned row = i / p.d_model;
    const unsigned feature = i % p.d_model;
    const unsigned source = head_major_index(row, feature, p);
    float *destination = projected_grad + row * 3 * p.d_model;
    destination[feature] += query_grad[source];
    destination[p.d_model + feature] += key_grad[source];
    destination[2 * p.d_model + feature] += value_grad[source];
}

__global__ void merge_heads_forward(const float *attended, float *merged,
                                    KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        merged[i] = attended[head_major_index(i / p.d_model, i % p.d_model, p)];
    }
}

__global__ void merge_heads_backward(float *attended_grad,
                                     const float *merged_grad,
                                     KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        attended_grad[head_major_index(i / p.d_model, i % p.d_model, p)] +=
            merged_grad[i];
    }
}

__global__ void attention_forward(const float *query, const float *key,
                                  const float *value, float *probabilities,
                                  float *output, KernelParameters p) {
    const unsigned query_row = global_thread_index();
    if (query_row >= p.rows) {
        return;
    }
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = (query_row / length) * length;
    const float *query_values = query + query_row * head_dimension;
    float *probability_row = probabilities + query_row * length;
    float maximum = -INFINITY;
    for (unsigned key_index = 0; key_index < length; ++key_index) {
        const float *key_values =
            key + (group_start + key_index) * head_dimension;
        float score = 0.0f;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            score += query_values[dimension] * key_values[dimension];
        }
        score *= p.scale;
        probability_row[key_index] = score;
        maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0f;
    for (unsigned key_index = 0; key_index < length; ++key_index) {
        probability_row[key_index] = expf(probability_row[key_index] - maximum);
        denominator += probability_row[key_index];
    }
    for (unsigned key_index = 0; key_index < length; ++key_index) {
        probability_row[key_index] /= denominator;
    }
    for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
        float result = 0.0f;
        for (unsigned key_index = 0; key_index < length; ++key_index) {
            result += probability_row[key_index] *
                      value[(group_start + key_index) * head_dimension +
                            dimension];
        }
        output[query_row * head_dimension + dimension] = result;
    }
}

__global__ void attention_backward_query(const float *key, const float *value,
                                         const float *probabilities,
                                         const float *output_grad,
                                         float *score_grad, float *query_grad,
                                         KernelParameters p) {
    const unsigned query_row = global_thread_index();
    if (query_row >= p.rows) {
        return;
    }
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = (query_row / length) * length;
    const float *upstream = output_grad + query_row * head_dimension;
    const float *probability_row = probabilities + query_row * length;
    float *score_row = score_grad + query_row * length;
    float probability_dot_gradient = 0.0f;
    for (unsigned key_index = 0; key_index < length; ++key_index) {
        const float *value_values =
            value + (group_start + key_index) * head_dimension;
        float probability_gradient = 0.0f;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            probability_gradient += upstream[dimension] * value_values[dimension];
        }
        score_row[key_index] = probability_gradient;
        probability_dot_gradient +=
            probability_row[key_index] * probability_gradient;
    }
    for (unsigned key_index = 0; key_index < length; ++key_index) {
        const float score_gradient = p.scale * probability_row[key_index] *
            (score_row[key_index] - probability_dot_gradient);
        score_row[key_index] = score_gradient;
        const float *key_values =
            key + (group_start + key_index) * head_dimension;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            query_grad[query_row * head_dimension + dimension] +=
                score_gradient * key_values[dimension];
        }
    }
}

__global__ void attention_backward_key_value(const float *query,
                                             const float *probabilities,
                                             const float *output_grad,
                                             const float *score_grad,
                                             float *key_grad,
                                             float *value_grad,
                                             KernelParameters p) {
    const unsigned key_row = global_thread_index();
    if (key_row >= p.rows) {
        return;
    }
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = (key_row / length) * length;
    const unsigned key_index = key_row - group_start;
    for (unsigned query_index = 0; query_index < length; ++query_index) {
        const unsigned query_row = group_start + query_index;
        const float score_gradient = score_grad[query_row * length + key_index];
        const float probability = probabilities[query_row * length + key_index];
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            key_grad[key_row * head_dimension + dimension] +=
                score_gradient * query[query_row * head_dimension + dimension];
            value_grad[key_row * head_dimension + dimension] +=
                probability * output_grad[query_row * head_dimension + dimension];
        }
    }
}

__global__ void attention_forward_shared(const float *query, const float *key,
                                         const float *value,
                                         float *probabilities, float *output,
                                         KernelParameters p) {
    extern __shared__ float shared[];
    const unsigned group = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned lanes = blockDim.x;
    const unsigned warp_lane = lane % kWarpSize;
    const unsigned warp = lane / kWarpSize;
    const unsigned warps = lanes / kWarpSize;
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_elements = length * head_dimension;
    const unsigned score_count = length * length;
    float *shared_queries = shared;
    float *shared_keys = shared_queries + group_elements;
    float *shared_values = shared_keys + group_elements;
    float *shared_scores = shared_values + group_elements;
    const unsigned group_offset = group * group_elements;
    for (unsigned i = lane; i < group_elements; i += lanes) {
        shared_queries[i] = query[group_offset + i];
        shared_keys[i] = key[group_offset + i];
        shared_values[i] = value[group_offset + i];
    }
    __syncthreads();

    for (unsigned pair = lane; pair < score_count; pair += lanes) {
        const unsigned query_index = pair / length;
        const unsigned key_index = pair % length;
        float score = 0.0f;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            score += shared_queries[query_index * head_dimension + dimension] *
                     shared_keys[key_index * head_dimension + dimension];
        }
        shared_scores[pair] = score * p.scale;
    }
    __syncthreads();

    for (unsigned query_index = warp; query_index < length;
         query_index += warps) {
        float *score_row = shared_scores + query_index * length;
        float maximum = -INFINITY;
        for (unsigned key_index = warp_lane; key_index < length;
             key_index += kWarpSize) {
            maximum = fmaxf(maximum, score_row[key_index]);
        }
        maximum = warp_max(maximum);
        float denominator = 0.0f;
        for (unsigned key_index = warp_lane; key_index < length;
             key_index += kWarpSize) {
            score_row[key_index] = expf(score_row[key_index] - maximum);
            denominator += score_row[key_index];
        }
        denominator = warp_sum(denominator);
        float *probability_output =
            probabilities + (group * length + query_index) * length;
        for (unsigned key_index = warp_lane; key_index < length;
             key_index += kWarpSize) {
            score_row[key_index] /= denominator;
            probability_output[key_index] = score_row[key_index];
        }
    }
    __syncthreads();

    for (unsigned element = lane; element < group_elements; element += lanes) {
        const unsigned query_index = element / head_dimension;
        const unsigned dimension = element % head_dimension;
        float result = 0.0f;
        for (unsigned key_index = 0; key_index < length; ++key_index) {
            result += shared_scores[query_index * length + key_index] *
                      shared_values[key_index * head_dimension + dimension];
        }
        output[group_offset + element] = result;
    }
}

__global__ void attention_backward_shared(
    const float *query, const float *key, const float *value,
    const float *probabilities, const float *output, const float *output_grad,
    float *query_grad, float *key_grad, float *value_grad,
    KernelParameters p) {
    extern __shared__ float shared[];
    const unsigned group = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned lanes = blockDim.x;
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_elements = length * head_dimension;
    const unsigned score_count = length * length;
    float *shared_queries = shared;
    float *shared_keys = shared_queries + group_elements;
    float *shared_values = shared_keys + group_elements;
    float *shared_upstream = shared_values + group_elements;
    float *shared_scores = shared_upstream + group_elements;
    float *upstream_dot_output = shared_scores + score_count;
    const unsigned group_offset = group * group_elements;
    for (unsigned i = lane; i < group_elements; i += lanes) {
        shared_queries[i] = query[group_offset + i];
        shared_keys[i] = key[group_offset + i];
        shared_values[i] = value[group_offset + i];
        shared_upstream[i] = output_grad[group_offset + i];
    }
    for (unsigned i = lane; i < score_count; i += lanes) {
        shared_scores[i] = probabilities[group * score_count + i];
    }
    __syncthreads();

    for (unsigned query_index = lane; query_index < length;
         query_index += lanes) {
        float dot = 0.0f;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            const unsigned element = query_index * head_dimension + dimension;
            dot += shared_upstream[element] * output[group_offset + element];
        }
        upstream_dot_output[query_index] = dot;
    }
    for (unsigned element = lane; element < group_elements; element += lanes) {
        const unsigned key_index = element / head_dimension;
        const unsigned dimension = element % head_dimension;
        float gradient = 0.0f;
        for (unsigned query_index = 0; query_index < length; ++query_index) {
            gradient += shared_scores[query_index * length + key_index] *
                        shared_upstream[query_index * head_dimension +
                                        dimension];
        }
        value_grad[group_offset + element] += gradient;
    }
    __syncthreads();

    for (unsigned pair = lane; pair < score_count; pair += lanes) {
        const unsigned query_index = pair / length;
        const unsigned key_index = pair % length;
        float probability_gradient = 0.0f;
        for (unsigned dimension = 0; dimension < head_dimension; ++dimension) {
            probability_gradient +=
                shared_upstream[query_index * head_dimension + dimension] *
                shared_values[key_index * head_dimension + dimension];
        }
        shared_scores[pair] = p.scale * shared_scores[pair] *
            (probability_gradient - upstream_dot_output[query_index]);
    }
    __syncthreads();

    for (unsigned work = lane; work < 2 * group_elements; work += lanes) {
        const unsigned element = work % group_elements;
        const unsigned row = element / head_dimension;
        const unsigned dimension = element % head_dimension;
        float gradient = 0.0f;
        if (work < group_elements) {
            for (unsigned key_index = 0; key_index < length; ++key_index) {
                gradient += shared_scores[row * length + key_index] *
                            shared_keys[key_index * head_dimension + dimension];
            }
            query_grad[group_offset + element] += gradient;
        } else {
            for (unsigned query_index = 0; query_index < length;
                 ++query_index) {
                gradient += shared_scores[query_index * length + row] *
                            shared_queries[query_index * head_dimension +
                                           dimension];
            }
            key_grad[group_offset + element] += gradient;
        }
    }
}

__device__ float2 combine_softmax_statistics(float2 first, float2 second) {
    if (second.x == -INFINITY) {
        return first;
    }
    if (first.x == -INFINITY) {
        return second;
    }
    const float maximum = fmaxf(first.x, second.x);
    return make_float2(maximum, first.y * expf(first.x - maximum) +
                                    second.y * expf(second.x - maximum));
}

__device__ float2 reduce_softmax_statistics_across_lanes(float2 statistics) {
    for (unsigned offset = kFlashLanesPerItem / 2; offset > 0; offset /= 2) {
        const float2 other = make_float2(
            __shfl_xor_sync(kFullWarpMask, statistics.x, offset),
            __shfl_xor_sync(kFullWarpMask, statistics.y, offset));
        statistics = combine_softmax_statistics(statistics, other);
    }
    return statistics;
}

__device__ float reduce_sum_across_item_lanes(float value) {
    for (unsigned offset = kFlashLanesPerItem / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ void load_tile(const float *source, float *tile,
                          unsigned first_row, unsigned row_count,
                          unsigned head_dimension) {
    for (unsigned i = threadIdx.x; i < kFlashBlockItems * head_dimension;
         i += blockDim.x) {
        const unsigned row = i / head_dimension;
        tile[i] = row < row_count
            ? source[(first_row + row) * head_dimension + i % head_dimension]
            : 0.0f;
    }
}

__global__ void attention_forward_flash(const float *query, const float *key,
                                        const float *value,
                                        float *probabilities, float *output,
                                        KernelParameters p) {
    __shared__ float key_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    __shared__ float value_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = blockIdx.y * length;
    const unsigned query_index =
        blockIdx.x * kFlashBlockItems + threadIdx.x / kFlashLanesPerItem;
    const unsigned lane = threadIdx.x % kFlashLanesPerItem;
    const bool active = query_index < length;
    float query_values[kFlashMaxHeadDimension];
    for (unsigned d = 0; d < head_dimension; ++d) {
        query_values[d] = active
            ? query[(group_start + query_index) * head_dimension + d]
            : 0.0f;
    }

    float2 statistics = make_float2(-INFINITY, 0.0f);
    for (unsigned tile_start = 0; tile_start < length;
         tile_start += kFlashBlockItems) {
        const unsigned tile_rows = min(kFlashBlockItems, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        __syncthreads();
        for (unsigned k = lane; active && k < tile_rows;
             k += kFlashLanesPerItem) {
            float score = 0.0f;
            for (unsigned d = 0; d < head_dimension; ++d) {
                score += query_values[d] * key_tile[k * head_dimension + d];
            }
            statistics = combine_softmax_statistics(
                statistics, make_float2(score * p.scale, 1.0f));
        }
        __syncthreads();
    }
    statistics = reduce_softmax_statistics_across_lanes(statistics);

    float output_values[kFlashMaxHeadDimension];
    for (unsigned d = 0; d < head_dimension; ++d) {
        output_values[d] = 0.0f;
    }
    float *probability_row =
        probabilities + (group_start + query_index) * length;
    for (unsigned tile_start = 0; tile_start < length;
         tile_start += kFlashBlockItems) {
        const unsigned tile_rows = min(kFlashBlockItems, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        load_tile(value, value_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        __syncthreads();
        for (unsigned k = lane; active && k < tile_rows;
             k += kFlashLanesPerItem) {
            float score = 0.0f;
            for (unsigned d = 0; d < head_dimension; ++d) {
                score += query_values[d] * key_tile[k * head_dimension + d];
            }
            const float probability =
                expf(score * p.scale - statistics.x) / statistics.y;
            probability_row[tile_start + k] = probability;
            for (unsigned d = 0; d < head_dimension; ++d) {
                output_values[d] +=
                    probability * value_tile[k * head_dimension + d];
            }
        }
        __syncthreads();
    }
    for (unsigned d = 0; d < head_dimension; ++d) {
        const float total = reduce_sum_across_item_lanes(output_values[d]);
        if (active && lane == 0) {
            output[(group_start + query_index) * head_dimension + d] = total;
        }
    }
}

__global__ void attention_backward_flash_queries(
    const float *key, const float *value, const float *probabilities,
    const float *output, const float *output_grad, float *score_grad,
    float *query_grad, KernelParameters p) {
    __shared__ float key_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    __shared__ float value_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = blockIdx.y * length;
    const unsigned query_index =
        blockIdx.x * kFlashBlockItems + threadIdx.x / kFlashLanesPerItem;
    const unsigned lane = threadIdx.x % kFlashLanesPerItem;
    const bool active = query_index < length;
    const unsigned query_row = group_start + query_index;
    float upstream[kFlashMaxHeadDimension];
    float upstream_dot_output = 0.0f;
    for (unsigned d = 0; d < head_dimension; ++d) {
        upstream[d] = active ? output_grad[query_row * head_dimension + d]
                             : 0.0f;
        upstream_dot_output += active
            ? upstream[d] * output[query_row * head_dimension + d]
            : 0.0f;
    }
    float query_gradient[kFlashMaxHeadDimension];
    for (unsigned d = 0; d < head_dimension; ++d) {
        query_gradient[d] = 0.0f;
    }
    for (unsigned tile_start = 0; tile_start < length;
         tile_start += kFlashBlockItems) {
        const unsigned tile_rows = min(kFlashBlockItems, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        load_tile(value, value_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        __syncthreads();
        for (unsigned k = lane; active && k < tile_rows;
             k += kFlashLanesPerItem) {
            float probability_gradient = 0.0f;
            for (unsigned d = 0; d < head_dimension; ++d) {
                probability_gradient +=
                    upstream[d] * value_tile[k * head_dimension + d];
            }
            const unsigned score_index = query_row * length + tile_start + k;
            const float score_gradient = p.scale * probabilities[score_index] *
                (probability_gradient - upstream_dot_output);
            score_grad[score_index] = score_gradient;
            for (unsigned d = 0; d < head_dimension; ++d) {
                query_gradient[d] +=
                    score_gradient * key_tile[k * head_dimension + d];
            }
        }
        __syncthreads();
    }
    for (unsigned d = 0; d < head_dimension; ++d) {
        const float total = reduce_sum_across_item_lanes(query_gradient[d]);
        if (active && lane == 0) {
            query_grad[query_row * head_dimension + d] += total;
        }
    }
}

__global__ void attention_backward_flash_keys(
    const float *query, const float *probabilities, const float *output_grad,
    const float *score_grad, float *key_grad, float *value_grad,
    KernelParameters p) {
    __shared__ float query_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    __shared__ float upstream_tile[kFlashBlockItems * kFlashMaxHeadDimension];
    const unsigned length = p.sequence_length;
    const unsigned head_dimension = p.cols;
    const unsigned group_start = blockIdx.y * length;
    const unsigned key_index =
        blockIdx.x * kFlashBlockItems + threadIdx.x / kFlashLanesPerItem;
    const unsigned lane = threadIdx.x % kFlashLanesPerItem;
    const bool active = key_index < length;
    float key_gradient[kFlashMaxHeadDimension];
    float value_gradient[kFlashMaxHeadDimension];
    for (unsigned d = 0; d < head_dimension; ++d) {
        key_gradient[d] = 0.0f;
        value_gradient[d] = 0.0f;
    }
    for (unsigned tile_start = 0; tile_start < length;
         tile_start += kFlashBlockItems) {
        const unsigned tile_rows = min(kFlashBlockItems, length - tile_start);
        load_tile(query, query_tile, group_start + tile_start, tile_rows,
                  head_dimension);
        load_tile(output_grad, upstream_tile, group_start + tile_start,
                  tile_rows, head_dimension);
        __syncthreads();
        for (unsigned q = lane; active && q < tile_rows;
             q += kFlashLanesPerItem) {
            const unsigned score_index =
                (group_start + tile_start + q) * length + key_index;
            const float probability = probabilities[score_index];
            const float score_gradient = score_grad[score_index];
            for (unsigned d = 0; d < head_dimension; ++d) {
                key_gradient[d] +=
                    score_gradient * query_tile[q * head_dimension + d];
                value_gradient[d] +=
                    probability * upstream_tile[q * head_dimension + d];
            }
        }
        __syncthreads();
    }
    const unsigned key_row = group_start + key_index;
    for (unsigned d = 0; d < head_dimension; ++d) {
        const float key_total = reduce_sum_across_item_lanes(key_gradient[d]);
        const float value_total =
            reduce_sum_across_item_lanes(value_gradient[d]);
        if (active && lane == 0) {
            key_grad[key_row * head_dimension + d] += key_total;
            value_grad[key_row * head_dimension + d] += value_total;
        }
    }
}

__global__ void zero_gradient(float *gradient, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        gradient[i] = 0.0f;
    }
}

__global__ void extract_patches(const float *images, float *patches,
                                KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned patch_row = i / p.cols;
    const unsigned element = i % p.cols;
    const unsigned patches_per_row = p.width / p.patch_size;
    const unsigned patches_per_image =
        (p.height / p.patch_size) * patches_per_row;
    const unsigned sample = patch_row / patches_per_image;
    const unsigned patch_index = patch_row % patches_per_image;
    const unsigned channel = element / (p.patch_size * p.patch_size);
    const unsigned y = (patch_index / patches_per_row) * p.patch_size +
                       (element / p.patch_size) % p.patch_size;
    const unsigned x = (patch_index % patches_per_row) * p.patch_size +
                       element % p.patch_size;
    patches[i] = images[((sample * p.channels + channel) * p.height + y) *
                        p.width + x];
}

__global__ void token_embedding_forward(const float *patch_tokens,
                                        const float *cls_token,
                                        const float *positional,
                                        float *tokens, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned dimension = i % p.d_model;
    const unsigned position = (i / p.d_model) % p.sequence_length;
    const unsigned sample = i / (p.d_model * p.sequence_length);
    const unsigned patch_count = p.sequence_length - 1;
    const float token = position == 0
        ? cls_token[dimension]
        : patch_tokens[(sample * patch_count + position - 1) * p.d_model +
                       dimension];
    tokens[i] = token + positional[position * p.d_model + dimension];
}

__global__ void token_embedding_backward_patches(float *patch_grad,
                                                 const float *tokens_grad,
                                                 KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned patch_count = p.sequence_length - 1;
    const unsigned dimension = i % p.d_model;
    const unsigned patch = (i / p.d_model) % patch_count;
    const unsigned sample = i / (p.d_model * patch_count);
    patch_grad[i] += tokens_grad[(sample * p.sequence_length + patch + 1) *
                                 p.d_model + dimension];
}

__global__ void token_embedding_backward_parameters(float *cls_grad,
                                                    float *positional_grad,
                                                    const float *tokens_grad,
                                                    KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const unsigned dimension = i % p.d_model;
    const unsigned position = i / p.d_model;
    float sum = 0.0f;
    for (unsigned sample = 0; sample < p.batch; ++sample) {
        sum += tokens_grad[(sample * p.sequence_length + position) * p.d_model +
                           dimension];
    }
    positional_grad[i] += sum;
    if (position == 0) {
        cls_grad[dimension] += sum;
    }
}

__global__ void gather_rows_forward(const float *source, float *destination,
                                    KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        destination[i] =
            source[(i / p.cols) * p.sequence_length * p.cols + i % p.cols];
    }
}

__global__ void gather_rows_backward(float *source_grad,
                                     const float *destination_grad,
                                     KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i < p.count) {
        source_grad[(i / p.cols) * p.sequence_length * p.cols + i % p.cols] +=
            destination_grad[i];
    }
}

__global__ void softmax_cross_entropy_rows(const float *logits,
                                           const float *targets,
                                           const float *class_weights,
                                           float *logits_grad,
                                           float *row_losses,
                                           KernelParameters p) {
    const unsigned row = global_thread_index();
    if (row >= p.rows) {
        return;
    }
    const float *row_logits = logits + row * p.cols;
    const unsigned target = static_cast<unsigned>(targets[row]);
    float maximum = -INFINITY;
    for (unsigned col = 0; col < p.cols; ++col) {
        maximum = fmaxf(maximum, row_logits[col]);
    }
    float denominator = 0.0f;
    for (unsigned col = 0; col < p.cols; ++col) {
        denominator += expf(row_logits[col] - maximum);
    }
    const float log_denominator = maximum + logf(denominator);
    const float weight = p.has_class_weights ? class_weights[target] : 1.0f;
    const float off_target_probability =
        p.label_smoothing / static_cast<float>(p.cols);
    const float on_target_probability =
        1.0f - p.label_smoothing + off_target_probability;
    float loss = 0.0f;
    for (unsigned col = 0; col < p.cols; ++col) {
        const float target_probability =
            col == target ? on_target_probability : off_target_probability;
        const float probability = expf(row_logits[col] - log_denominator);
        loss += weight * target_probability *
                (log_denominator - row_logits[col]);
        logits_grad[row * p.cols + col] +=
            weight * (probability - target_probability) /
            static_cast<float>(p.rows);
    }
    row_losses[row] = loss;
}

__global__ void mean_loss(const float *row_losses, float *loss,
                          KernelParameters p) {
    if (global_thread_index() != 0) {
        return;
    }
    float sum = 0.0f;
    for (unsigned row = 0; row < p.rows; ++row) {
        sum += row_losses[row];
    }
    loss[0] = sum / static_cast<float>(p.rows);
}

__global__ void clip_scale(const float *partials, float *scale,
                           KernelParameters p) {
    if (global_thread_index() != 0) {
        return;
    }
    float squared_norm = 0.0f;
    for (unsigned index = 0; index < p.count; ++index) {
        squared_norm += partials[index];
    }
    const float norm = sqrtf(squared_norm);
    scale[0] = norm > p.max_norm && norm > 0.0f ? p.max_norm / norm : 1.0f;
}

__global__ void adamw_step(float *value, float *gradient, float *first_moment,
                           float *second_moment, KernelParameters p) {
    const unsigned i = global_thread_index();
    if (i >= p.count) {
        return;
    }
    const float g = gradient[i];
    first_moment[i] = p.beta1 * first_moment[i] + (1.0f - p.beta1) * g;
    second_moment[i] = p.beta2 * second_moment[i] + (1.0f - p.beta2) * g * g;
    const float estimate = (first_moment[i] / p.correction1) /
        (sqrtf(second_moment[i] / p.correction2) + p.epsilon);
    value[i] -= p.learning_rate * (estimate + p.weight_decay * value[i]);
    gradient[i] = 0.0f;
}

__global__ void column_sums_tiled(const float *values, float *column_sums,
                                  KernelParameters p) {
    extern __shared__ float partial_sums[];
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned slot = threadIdx.y * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (col < p.cols) {
        for (unsigned row = threadIdx.y; row < p.rows; row += blockDim.y) {
            sum += values[row * p.cols + col];
        }
    }
    partial_sums[slot] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            partial_sums[slot] += partial_sums[slot + stride * blockDim.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && col < p.cols) {
        column_sums[col] += partial_sums[threadIdx.x];
    }
}

__device__ void warp_row_statistics(const float *row_values, unsigned cols,
                                    float epsilon, unsigned lane, float &mean,
                                    float &inverse_std) {
    float sum = 0.0f;
    for (unsigned col = lane; col < cols; col += kWarpSize) {
        sum += row_values[col];
    }
    mean = warp_sum(sum) / static_cast<float>(cols);
    float variance = 0.0f;
    for (unsigned col = lane; col < cols; col += kWarpSize) {
        const float centered = row_values[col] - mean;
        variance += centered * centered;
    }
    inverse_std = 1.0f / sqrtf(warp_sum(variance) / static_cast<float>(cols) +
                               epsilon);
}

__global__ void layer_norm_forward_warp(const float *input, const float *gamma,
                                        const float *beta, float *output,
                                        KernelParameters p) {
    const unsigned row = global_thread_index() / kWarpSize;
    const unsigned lane = threadIdx.x % kWarpSize;
    if (row >= p.rows) {
        return;
    }
    const float *row_values = input + row * p.cols;
    float mean;
    float inverse_std;
    warp_row_statistics(row_values, p.cols, p.epsilon, lane, mean,
                        inverse_std);
    for (unsigned col = lane; col < p.cols; col += kWarpSize) {
        output[row * p.cols + col] =
            (row_values[col] - mean) * inverse_std * gamma[col] + beta[col];
    }
}

__global__ void layer_norm_backward_rows_warp(const float *input,
                                              const float *gamma,
                                              const float *output_grad,
                                              float *input_grad,
                                              float *statistics,
                                              KernelParameters p) {
    const unsigned row = global_thread_index() / kWarpSize;
    const unsigned lane = threadIdx.x % kWarpSize;
    if (row >= p.rows) {
        return;
    }
    const float *row_values = input + row * p.cols;
    const float *upstream = output_grad + row * p.cols;
    float mean;
    float inverse_std;
    warp_row_statistics(row_values, p.cols, p.epsilon, lane, mean,
                        inverse_std);
    if (lane == 0) {
        statistics[2 * row] = mean;
        statistics[2 * row + 1] = inverse_std;
    }
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (unsigned col = lane; col < p.cols; col += kWarpSize) {
        const float normalized = (row_values[col] - mean) * inverse_std;
        const float dxhat = upstream[col] * gamma[col];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * normalized;
    }
    const float cols = static_cast<float>(p.cols);
    const float mean_dxhat = warp_sum(sum_dxhat) / cols;
    const float mean_dxhat_xhat = warp_sum(sum_dxhat_xhat) / cols;
    for (unsigned col = lane; col < p.cols; col += kWarpSize) {
        const float normalized = (row_values[col] - mean) * inverse_std;
        const float dxhat = upstream[col] * gamma[col];
        input_grad[row * p.cols + col] +=
            inverse_std * (dxhat - mean_dxhat - normalized * mean_dxhat_xhat);
    }
}

__global__ void layer_norm_parameter_gradients_tiled(
    const float *input, const float *output_grad, const float *statistics,
    float *gamma_grad, float *beta_grad, KernelParameters p) {
    extern __shared__ float partial_sums[];
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tile_size = blockDim.x * blockDim.y;
    const unsigned slot = threadIdx.y * blockDim.x + threadIdx.x;
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    if (col < p.cols) {
        for (unsigned row = threadIdx.y; row < p.rows; row += blockDim.y) {
            const float upstream = output_grad[row * p.cols + col];
            const float normalized =
                (input[row * p.cols + col] - statistics[2 * row]) *
                statistics[2 * row + 1];
            gamma_sum += upstream * normalized;
            beta_sum += upstream;
        }
    }
    partial_sums[slot] = gamma_sum;
    partial_sums[tile_size + slot] = beta_sum;
    __syncthreads();
    for (unsigned stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            partial_sums[slot] += partial_sums[slot + stride * blockDim.x];
            partial_sums[tile_size + slot] +=
                partial_sums[tile_size + slot + stride * blockDim.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && col < p.cols) {
        gamma_grad[col] += partial_sums[threadIdx.x];
        beta_grad[col] += partial_sums[tile_size + threadIdx.x];
    }
}

__device__ unsigned find_view(const ParameterView *views, unsigned view_count,
                              unsigned element) {
    unsigned low = 0;
    unsigned high = view_count - 1;
    while (low < high) {
        const unsigned middle = (low + high + 1) / 2;
        if (views[middle].element_offset <= element) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

__global__ void zero_gradients_views(const ParameterView *views,
                                     KernelParameters p) {
    const unsigned element = global_thread_index();
    if (element >= p.count) {
        return;
    }
    const unsigned view = find_view(views, p.parameter_index, element);
    views[view].gradient[element - views[view].element_offset] = 0.0f;
}

__global__ void squared_norm_views(const ParameterView *views,
                                   float *partials) {
    __shared__ float shared_sums[kThreadsPerBlock];
    const unsigned view = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const float *gradient = views[view].gradient;
    float sum = 0.0f;
    for (unsigned i = lane; i < views[view].count; i += blockDim.x) {
        sum += gradient[i] * gradient[i];
    }
    shared_sums[lane] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        partials[view] = shared_sums[0];
    }
}

__global__ void scale_gradients_views(const ParameterView *views,
                                      const float *scale, KernelParameters p) {
    const unsigned element = global_thread_index();
    if (element >= p.count) {
        return;
    }
    const unsigned view = find_view(views, p.parameter_index, element);
    views[view].gradient[element - views[view].element_offset] *= scale[0];
}

__global__ void adamw_views(const ParameterView *views,
                            const float *gradient_scale, KernelParameters p) {
    const unsigned element = global_thread_index();
    if (element >= p.count) {
        return;
    }
    const ParameterView &view =
        views[find_view(views, p.parameter_index, element)];
    const unsigned i = element - view.element_offset;
    const float g = p.max_norm > 0.0f ? view.gradient[i] * gradient_scale[0]
                                      : view.gradient[i];
    const float first = p.beta1 * view.first_moment[i] + (1.0f - p.beta1) * g;
    const float second = p.beta2 * view.second_moment[i] +
                         (1.0f - p.beta2) * g * g;
    view.first_moment[i] = first;
    view.second_moment[i] = second;
    const float estimate = (first / view.correction1) /
        (sqrtf(second / view.correction2) + p.epsilon);
    view.value[i] -= p.learning_rate *
                     (estimate + view.weight_decay * view.value[i]);
    view.gradient[i] = 0.0f;
}

void abort_on_cuda_error(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation,
                cudaGetErrorString(status));
        abort();
    }
}

std::unordered_set<const void *> &managed_allocations() {
    static std::unordered_set<const void *> allocations;
    return allocations;
}

bool is_device_memory(const float *memory) {
    return memory && managed_allocations().count(memory) != 0;
}

bool all_device_memory(std::initializer_list<const float *> memories) {
    for (const float *memory : memories) {
        if (!is_device_memory(memory)) {
            return false;
        }
    }
    return true;
}

bool fits_kernel_index(size_t value) {
    return value <= UINT_MAX - kThreadsPerBlock;
}

bool fits_blas_dimension(size_t value) {
    return value > 0 && value <= INT_MAX;
}

unsigned blocks_for(size_t threads, unsigned threads_per_block) {
    return static_cast<unsigned>((threads + threads_per_block - 1) /
                                 threads_per_block);
}

struct DeviceScratch {
    void *memory = nullptr;
    size_t capacity_bytes = 0;
};

}

struct CudaBackend {
    int device;
    cudaStream_t stream;
    cublasHandle_t blas;
    size_t max_shared_memory_per_block;
    DeviceScratch layer_norm_statistics;
    DeviceScratch row_losses;
    DeviceScratch gradient_norm_partials;
    DeviceScratch gradient_scale;
    DeviceScratch parameter_views;
    OpsDeviceBackend device_backend;
    int device_execution_enabled;
};

namespace {

template <typename T>
T *scratch_memory(DeviceScratch &scratch, size_t count) {
    const size_t required_bytes = (count > 0 ? count : 1) * sizeof(T);
    if (scratch.capacity_bytes < required_bytes) {
        if (scratch.memory) {
            abort_on_cuda_error(cudaFree(scratch.memory), "cudaFree");
        }
        abort_on_cuda_error(cudaMalloc(&scratch.memory, required_bytes),
                            "cudaMalloc");
        scratch.capacity_bytes = required_bytes;
    }
    return static_cast<T *>(scratch.memory);
}

void release_scratch(DeviceScratch &scratch) {
    if (scratch.memory) {
        cudaFree(scratch.memory);
    }
    scratch = DeviceScratch{};
}

void check_launch(const char *kernel_name) {
    abort_on_cuda_error(cudaGetLastError(), kernel_name);
}

template <typename Kernel, typename... Arguments>
void launch_elementwise(CudaBackend *backend, const char *name, Kernel kernel,
                        size_t threads, Arguments... arguments) {
    kernel<<<blocks_for(threads, kThreadsPerBlock), kThreadsPerBlock, 0,
             backend->stream>>>(arguments...);
    check_launch(name);
}

KernelParameters element_parameters(const Tensor *tensor) {
    KernelParameters parameters = {};
    parameters.count = static_cast<unsigned>(tensor_numel(tensor));
    return parameters;
}

void *allocate_device_memory(void *context, size_t bytes) {
    (void)context;
    void *memory = nullptr;
    if (bytes == 0 ||
        cudaMallocManaged(&memory, bytes, cudaMemAttachGlobal) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    managed_allocations().insert(memory);
    return memory;
}

void release_device_memory(void *context, void *memory) {
    (void)context;
    if (memory && managed_allocations().erase(memory) != 0) {
        abort_on_cuda_error(cudaFree(memory), "cudaFree");
    }
}

void synchronize_device(void *context) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    abort_on_cuda_error(cudaStreamSynchronize(backend->stream),
                        "cudaStreamSynchronize");
}

int device_gemm(void *context, const Tensor *left, const Tensor *right,
                Tensor *output) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({left->data, right->data, output->data}) ||
        !fits_blas_dimension(left->rows) || !fits_blas_dimension(left->cols) ||
        !fits_blas_dimension(right->cols)) {
        return -1;
    }
    const int rows = static_cast<int>(left->rows);
    const int interior = static_cast<int>(left->cols);
    const int cols = static_cast<int>(right->cols);
    const float one = 1.0f;
    const float zero = 0.0f;
    return cublasSgemm(backend->blas, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows,
                       interior, &one, right->data, cols, left->data, interior,
                       &zero, output->data, cols) == CUBLAS_STATUS_SUCCESS
        ? 0
        : -1;
}

int device_gemm_backward(void *context, Tensor *left, Tensor *right,
                         const Tensor *output) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({left->data, left->grad, right->data, right->grad,
                            output->grad}) ||
        !fits_blas_dimension(left->rows) || !fits_blas_dimension(left->cols) ||
        !fits_blas_dimension(right->cols)) {
        return -1;
    }
    const int rows = static_cast<int>(left->rows);
    const int interior = static_cast<int>(left->cols);
    const int cols = static_cast<int>(right->cols);
    const float one = 1.0f;
    const cublasStatus_t left_status =
        cublasSgemm(backend->blas, CUBLAS_OP_T, CUBLAS_OP_N, interior, rows,
                    cols, &one, right->data, cols, output->grad, cols, &one,
                    left->grad, interior);
    if (left_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS left-gradient GEMM failed\n");
        abort();
    }
    const cublasStatus_t right_status =
        cublasSgemm(backend->blas, CUBLAS_OP_N, CUBLAS_OP_T, cols, interior,
                    rows, &one, output->grad, cols, left->data, interior, &one,
                    right->grad, cols);
    if (right_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS right-gradient GEMM failed\n");
        abort();
    }
    return 0;
}

int device_residual(void *context, const Tensor *left, const Tensor *right,
                    Tensor *output) {
    if (!all_device_memory({left->data, right->data, output->data}) ||
        !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context), "residual_forward",
                       residual_forward, tensor_numel(output), left->data,
                       right->data, output->data, element_parameters(output));
    return 0;
}

int device_residual_backward(void *context, Tensor *left, Tensor *right,
                             const Tensor *output) {
    if (!all_device_memory({left->grad, right->grad, output->grad}) ||
        left->grad == right->grad || !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "residual_backward", residual_backward,
                       tensor_numel(output), left->grad, right->grad,
                       output->grad, element_parameters(output));
    return 0;
}

int device_accumulate_gradient(void *context, Tensor *destination,
                               const Tensor *source) {
    if (!all_device_memory({destination->grad, source->grad}) ||
        !fits_kernel_index(tensor_numel(destination))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "accumulate_gradient", accumulate_gradient,
                       tensor_numel(destination), destination->grad,
                       source->grad, element_parameters(destination));
    return 0;
}

int device_bias_add(void *context, const Tensor *input, const Tensor *bias,
                    Tensor *output) {
    if (!all_device_memory({input->data, bias->data, output->data}) ||
        !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    KernelParameters parameters = element_parameters(output);
    parameters.cols = static_cast<unsigned>(output->cols);
    launch_elementwise(static_cast<CudaBackend *>(context), "bias_add_forward",
                       bias_add_forward, tensor_numel(output), input->data,
                       bias->data, output->data, parameters);
    return 0;
}

int device_bias_add_backward(void *context, const Tensor *output,
                             Tensor *bias_grad) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({output->grad, bias_grad->grad}) ||
        !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    KernelParameters parameters = {};
    parameters.rows = static_cast<unsigned>(output->rows);
    parameters.cols = static_cast<unsigned>(output->cols);
    const dim3 threads(kTileColumns, kTileRows);
    column_sums_tiled<<<blocks_for(output->cols, kTileColumns), threads,
                        kTileColumns * kTileRows * sizeof(float),
                        backend->stream>>>(output->grad, bias_grad->grad,
                                           parameters);
    check_launch("column_sums_tiled");
    return 0;
}

int device_gelu(void *context, const Tensor *input, Tensor *output) {
    if (!all_device_memory({input->data, output->data}) ||
        !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context), "gelu_forward",
                       gelu_forward, tensor_numel(output), input->data,
                       output->data, element_parameters(output));
    return 0;
}

int device_gelu_backward(void *context, const Tensor *input,
                         const Tensor *output) {
    if (!all_device_memory({input->data, input->grad, output->grad}) ||
        !fits_kernel_index(tensor_numel(output))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context), "gelu_backward",
                       gelu_backward, tensor_numel(output), input->data,
                       input->grad, output->grad, element_parameters(output));
    return 0;
}

int device_layer_norm(void *context, const Tensor *input, const Tensor *gamma,
                      const Tensor *beta, float epsilon, Tensor *output) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({input->data, gamma->data, beta->data,
                            output->data}) ||
        !fits_kernel_index(input->rows * kWarpSize)) {
        return -1;
    }
    KernelParameters parameters = {};
    parameters.rows = static_cast<unsigned>(input->rows);
    parameters.cols = static_cast<unsigned>(input->cols);
    parameters.epsilon = epsilon;
    layer_norm_forward_warp<<<blocks_for(input->rows, kRowsPerLayerNormBlock),
                              kThreadsPerBlock, 0, backend->stream>>>(
        input->data, gamma->data, beta->data, output->data, parameters);
    check_launch("layer_norm_forward_warp");
    return 0;
}

int device_layer_norm_backward(void *context, const Tensor *input,
                               const Tensor *gamma, float epsilon,
                               const Tensor *output, Tensor *input_grad,
                               Tensor *gamma_grad, Tensor *beta_grad) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({input->data, gamma->data, output->grad,
                            input_grad->grad, gamma_grad->grad,
                            beta_grad->grad}) ||
        !fits_kernel_index(input->rows * kWarpSize)) {
        return -1;
    }
    float *statistics =
        scratch_memory<float>(backend->layer_norm_statistics, 2 * input->rows);
    KernelParameters parameters = {};
    parameters.rows = static_cast<unsigned>(input->rows);
    parameters.cols = static_cast<unsigned>(input->cols);
    parameters.epsilon = epsilon;
    layer_norm_backward_rows_warp<<<blocks_for(input->rows,
                                               kRowsPerLayerNormBlock),
                                    kThreadsPerBlock, 0, backend->stream>>>(
        input->data, gamma->data, output->grad, input_grad->grad, statistics,
        parameters);
    check_launch("layer_norm_backward_rows_warp");
    const size_t gamma_and_beta_sums = 2;
    const dim3 threads(kTileColumns, kTileRows);
    layer_norm_parameter_gradients_tiled<<<
        blocks_for(input->cols, kTileColumns), threads,
        gamma_and_beta_sums * kTileColumns * kTileRows * sizeof(float),
        backend->stream>>>(input->data, output->grad, statistics,
                           gamma_grad->grad, beta_grad->grad, parameters);
    check_launch("layer_norm_parameter_gradients_tiled");
    return 0;
}

KernelParameters head_parameters(size_t batch, size_t sequence_length,
                                 size_t d_model, size_t heads) {
    KernelParameters parameters = {};
    parameters.count =
        static_cast<unsigned>(batch * sequence_length * d_model);
    parameters.batch = static_cast<unsigned>(batch);
    parameters.sequence_length = static_cast<unsigned>(sequence_length);
    parameters.d_model = static_cast<unsigned>(d_model);
    parameters.heads = static_cast<unsigned>(heads);
    return parameters;
}

int device_split_heads(void *context, const Tensor *projected, Tensor *query,
                       Tensor *key, Tensor *value, size_t batch,
                       size_t sequence_length, size_t d_model, size_t heads) {
    if (!all_device_memory({projected->data, query->data, key->data,
                            value->data}) ||
        !fits_kernel_index(tensor_numel(projected))) {
        return -1;
    }
    const KernelParameters parameters =
        head_parameters(batch, sequence_length, d_model, heads);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "split_heads_forward", split_heads_forward,
                       parameters.count, projected->data, query->data,
                       key->data, value->data, parameters);
    return 0;
}

int device_split_heads_backward(void *context, Tensor *projected,
                                const Tensor *query, const Tensor *key,
                                const Tensor *value, size_t batch,
                                size_t sequence_length, size_t d_model,
                                size_t heads) {
    if (!all_device_memory({projected->grad, query->grad, key->grad,
                            value->grad}) ||
        !fits_kernel_index(tensor_numel(projected))) {
        return -1;
    }
    const KernelParameters parameters =
        head_parameters(batch, sequence_length, d_model, heads);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "split_heads_backward", split_heads_backward,
                       parameters.count, projected->grad, query->grad,
                       key->grad, value->grad, parameters);
    return 0;
}

int device_merge_heads(void *context, const Tensor *attended, Tensor *merged,
                       size_t batch, size_t sequence_length, size_t d_model,
                       size_t heads) {
    if (!all_device_memory({attended->data, merged->data}) ||
        !fits_kernel_index(tensor_numel(merged))) {
        return -1;
    }
    const KernelParameters parameters =
        head_parameters(batch, sequence_length, d_model, heads);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "merge_heads_forward", merge_heads_forward,
                       parameters.count, attended->data, merged->data,
                       parameters);
    return 0;
}

int device_merge_heads_backward(void *context, Tensor *attended,
                                const Tensor *merged, size_t batch,
                                size_t sequence_length, size_t d_model,
                                size_t heads) {
    if (!all_device_memory({attended->grad, merged->grad}) ||
        !fits_kernel_index(tensor_numel(merged))) {
        return -1;
    }
    const KernelParameters parameters =
        head_parameters(batch, sequence_length, d_model, heads);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "merge_heads_backward", merge_heads_backward,
                       parameters.count, attended->grad, merged->grad,
                       parameters);
    return 0;
}

KernelParameters attention_parameters(const Tensor *query,
                                      size_t sequence_length, float scale) {
    KernelParameters parameters = {};
    parameters.rows = static_cast<unsigned>(query->rows);
    parameters.cols = static_cast<unsigned>(query->cols);
    parameters.sequence_length = static_cast<unsigned>(sequence_length);
    parameters.scale = scale;
    return parameters;
}

size_t attention_forward_shared_bytes(size_t sequence_length,
                                      size_t head_dimension) {
    return (3 * sequence_length * head_dimension +
            sequence_length * sequence_length) * sizeof(float);
}

size_t attention_backward_shared_bytes(size_t sequence_length,
                                       size_t head_dimension) {
    return (4 * sequence_length * head_dimension +
            sequence_length * sequence_length + sequence_length) *
           sizeof(float);
}

dim3 flash_grid(size_t batch, size_t heads, size_t sequence_length) {
    return dim3(blocks_for(sequence_length, kFlashBlockItems),
                static_cast<unsigned>(batch * heads));
}

int device_multi_head_attention(void *context, const Tensor *query,
                                const Tensor *key, const Tensor *value,
                                size_t batch, size_t heads,
                                size_t sequence_length, float scale,
                                Tensor *probabilities, Tensor *output) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({query->data, key->data, value->data,
                            probabilities->data, output->data}) ||
        !fits_kernel_index(tensor_numel(probabilities))) {
        return -1;
    }
    const KernelParameters parameters =
        attention_parameters(query, sequence_length, scale);
    const size_t shared_bytes =
        attention_forward_shared_bytes(sequence_length, query->cols);
    if (shared_bytes <= backend->max_shared_memory_per_block) {
        attention_forward_shared<<<static_cast<unsigned>(batch * heads),
                                   kThreadsPerBlock, shared_bytes,
                                   backend->stream>>>(
            query->data, key->data, value->data, probabilities->data,
            output->data, parameters);
        check_launch("attention_forward_shared");
    } else if (query->cols <= kFlashMaxHeadDimension) {
        attention_forward_flash<<<flash_grid(batch, heads, sequence_length),
                                  kThreadsPerBlock, 0, backend->stream>>>(
            query->data, key->data, value->data, probabilities->data,
            output->data, parameters);
        check_launch("attention_forward_flash");
    } else {
        launch_elementwise(backend, "attention_forward", attention_forward,
                           query->rows, query->data, key->data, value->data,
                           probabilities->data, output->data, parameters);
    }
    return 0;
}

int device_multi_head_attention_backward(
    void *context, const Tensor *query, const Tensor *key, const Tensor *value,
    size_t batch, size_t heads, size_t sequence_length, float scale,
    const Tensor *probabilities, const Tensor *output, Tensor *query_grad,
    Tensor *key_grad, Tensor *value_grad) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({query->data, key->data, value->data,
                            probabilities->data, probabilities->grad,
                            output->data, output->grad, query_grad->grad,
                            key_grad->grad, value_grad->grad}) ||
        !fits_kernel_index(tensor_numel(probabilities))) {
        return -1;
    }
    const KernelParameters parameters =
        attention_parameters(query, sequence_length, scale);
    const size_t shared_bytes =
        attention_backward_shared_bytes(sequence_length, query->cols);
    if (shared_bytes <= backend->max_shared_memory_per_block) {
        attention_backward_shared<<<static_cast<unsigned>(batch * heads),
                                    kThreadsPerBlock, shared_bytes,
                                    backend->stream>>>(
            query->data, key->data, value->data, probabilities->data,
            output->data, output->grad, query_grad->grad, key_grad->grad,
            value_grad->grad, parameters);
        check_launch("attention_backward_shared");
    } else if (query->cols <= kFlashMaxHeadDimension) {
        const dim3 grid = flash_grid(batch, heads, sequence_length);
        attention_backward_flash_queries<<<grid, kThreadsPerBlock, 0,
                                           backend->stream>>>(
            key->data, value->data, probabilities->data, output->data,
            output->grad, probabilities->grad, query_grad->grad, parameters);
        check_launch("attention_backward_flash_queries");
        attention_backward_flash_keys<<<grid, kThreadsPerBlock, 0,
                                        backend->stream>>>(
            query->data, probabilities->data, output->grad,
            probabilities->grad, key_grad->grad, value_grad->grad, parameters);
        check_launch("attention_backward_flash_keys");
    } else {
        launch_elementwise(backend, "attention_backward_query",
                           attention_backward_query, query->rows, key->data,
                           value->data, probabilities->data, output->grad,
                           probabilities->grad, query_grad->grad, parameters);
        launch_elementwise(backend, "attention_backward_key_value",
                           attention_backward_key_value, query->rows,
                           query->data, probabilities->data, output->grad,
                           probabilities->grad, key_grad->grad,
                           value_grad->grad, parameters);
    }
    return 0;
}

int device_zero_gradient(void *context, Tensor *tensor) {
    if (!is_device_memory(tensor->grad) ||
        !fits_kernel_index(tensor_numel(tensor))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context), "zero_gradient",
                       zero_gradient, tensor_numel(tensor), tensor->grad,
                       element_parameters(tensor));
    return 0;
}

int device_extract_patches(void *context, const Tensor *images,
                           Tensor *patches, size_t channels, size_t height,
                           size_t width, size_t patch_size) {
    if (!all_device_memory({images->data, patches->data}) ||
        !fits_kernel_index(tensor_numel(patches)) ||
        !fits_kernel_index(tensor_numel(images))) {
        return -1;
    }
    KernelParameters parameters = element_parameters(patches);
    parameters.cols = static_cast<unsigned>(patches->cols);
    parameters.channels = static_cast<unsigned>(channels);
    parameters.height = static_cast<unsigned>(height);
    parameters.width = static_cast<unsigned>(width);
    parameters.patch_size = static_cast<unsigned>(patch_size);
    launch_elementwise(static_cast<CudaBackend *>(context), "extract_patches",
                       extract_patches, tensor_numel(patches), images->data,
                       patches->data, parameters);
    return 0;
}

KernelParameters token_parameters(const Tensor *tokens,
                                  const Tensor *positional, size_t batch) {
    KernelParameters parameters = element_parameters(tokens);
    parameters.batch = static_cast<unsigned>(batch);
    parameters.sequence_length = static_cast<unsigned>(positional->rows);
    parameters.d_model = static_cast<unsigned>(tokens->cols);
    return parameters;
}

int device_token_embedding(void *context, const Tensor *patch_tokens,
                           const Tensor *cls_token, const Tensor *positional,
                           Tensor *tokens, size_t batch) {
    if (!all_device_memory({patch_tokens->data, cls_token->data,
                            positional->data, tokens->data}) ||
        !fits_kernel_index(tensor_numel(tokens))) {
        return -1;
    }
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "token_embedding_forward", token_embedding_forward,
                       tensor_numel(tokens), patch_tokens->data,
                       cls_token->data, positional->data, tokens->data,
                       token_parameters(tokens, positional, batch));
    return 0;
}

int device_token_embedding_backward(void *context, Tensor *patch_tokens,
                                    Tensor *cls_token, Tensor *positional,
                                    const Tensor *tokens, size_t batch) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({patch_tokens->grad, cls_token->grad,
                            positional->grad, tokens->grad}) ||
        !fits_kernel_index(tensor_numel(tokens))) {
        return -1;
    }
    KernelParameters patch_parameters =
        token_parameters(tokens, positional, batch);
    patch_parameters.count = static_cast<unsigned>(tensor_numel(patch_tokens));
    KernelParameters parameter_parameters =
        token_parameters(tokens, positional, batch);
    parameter_parameters.count = static_cast<unsigned>(tensor_numel(positional));
    launch_elementwise(backend, "token_embedding_backward_patches",
                       token_embedding_backward_patches,
                       tensor_numel(patch_tokens), patch_tokens->grad,
                       tokens->grad, patch_parameters);
    launch_elementwise(backend, "token_embedding_backward_parameters",
                       token_embedding_backward_parameters,
                       tensor_numel(positional), cls_token->grad,
                       positional->grad, tokens->grad, parameter_parameters);
    return 0;
}

int device_gather_rows(void *context, const Tensor *source,
                       Tensor *destination, size_t row_stride) {
    if (!all_device_memory({source->data, destination->data}) ||
        !fits_kernel_index(tensor_numel(source))) {
        return -1;
    }
    KernelParameters parameters = element_parameters(destination);
    parameters.cols = static_cast<unsigned>(destination->cols);
    parameters.sequence_length = static_cast<unsigned>(row_stride);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "gather_rows_forward", gather_rows_forward,
                       tensor_numel(destination), source->data,
                       destination->data, parameters);
    return 0;
}

int device_gather_rows_backward(void *context, Tensor *source,
                                const Tensor *destination, size_t row_stride) {
    if (!all_device_memory({source->grad, destination->grad}) ||
        !fits_kernel_index(tensor_numel(source))) {
        return -1;
    }
    KernelParameters parameters = element_parameters(destination);
    parameters.cols = static_cast<unsigned>(destination->cols);
    parameters.sequence_length = static_cast<unsigned>(row_stride);
    launch_elementwise(static_cast<CudaBackend *>(context),
                       "gather_rows_backward", gather_rows_backward,
                       tensor_numel(destination), source->grad,
                       destination->grad, parameters);
    return 0;
}

int device_softmax_cross_entropy(void *context, const Tensor *logits,
                                 const Tensor *targets,
                                 const Tensor *class_weights,
                                 float label_smoothing, Tensor *loss,
                                 Tensor *logits_grad) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    if (!all_device_memory({logits->data, targets->data, loss->data,
                            logits_grad->grad}) ||
        (class_weights && !is_device_memory(class_weights->data)) ||
        !fits_kernel_index(tensor_numel(logits))) {
        return -1;
    }
    float *row_losses = scratch_memory<float>(backend->row_losses, logits->rows);
    KernelParameters parameters = {};
    parameters.rows = static_cast<unsigned>(logits->rows);
    parameters.cols = static_cast<unsigned>(logits->cols);
    parameters.has_class_weights = class_weights ? 1u : 0u;
    parameters.label_smoothing = label_smoothing;
    launch_elementwise(backend, "softmax_cross_entropy_rows",
                       softmax_cross_entropy_rows, logits->rows, logits->data,
                       targets->data,
                       class_weights ? class_weights->data : logits->data,
                       logits_grad->grad, row_losses, parameters);
    mean_loss<<<1, 1, 0, backend->stream>>>(row_losses, loss->data,
                                            parameters);
    check_launch("mean_loss");
    return 0;
}

int device_adamw_step(void *context, Parameter *parameter, float learning_rate,
                      float beta1, float beta2, float epsilon,
                      float weight_decay, float correction1,
                      float correction2) {
    if (!all_device_memory({parameter->value.data, parameter->value.grad,
                            parameter->m, parameter->v}) ||
        !fits_kernel_index(tensor_numel(&parameter->value))) {
        return -1;
    }
    KernelParameters parameters = element_parameters(&parameter->value);
    parameters.learning_rate = learning_rate;
    parameters.beta1 = beta1;
    parameters.beta2 = beta2;
    parameters.epsilon = epsilon;
    parameters.weight_decay = weight_decay;
    parameters.correction1 = correction1;
    parameters.correction2 = correction2;
    launch_elementwise(static_cast<CudaBackend *>(context), "adamw_step",
                       adamw_step, tensor_numel(&parameter->value),
                       parameter->value.data, parameter->value.grad,
                       parameter->m, parameter->v, parameters);
    return 0;
}

struct ParameterViewTable {
    const ParameterView *device_views = nullptr;
    unsigned view_count = 0;
    size_t element_count = 0;
};

int upload_view_table(CudaBackend *backend,
                      const std::vector<ParameterView> &views,
                      ParameterViewTable *table) {
    size_t element_count = 0;
    for (const ParameterView &view : views) {
        element_count += view.count;
    }
    if (views.empty() || element_count == 0 ||
        !fits_kernel_index(element_count) || !fits_kernel_index(views.size())) {
        return -1;
    }
    ParameterView *device_views = scratch_memory<ParameterView>(
        backend->parameter_views, views.size());
    abort_on_cuda_error(
        cudaMemcpyAsync(device_views, views.data(),
                        views.size() * sizeof(ParameterView),
                        cudaMemcpyHostToDevice, backend->stream),
        "cudaMemcpyAsync");
    table->device_views = device_views;
    table->view_count = static_cast<unsigned>(views.size());
    table->element_count = element_count;
    return 0;
}

int build_gradient_views(Tensor *const *tensors, size_t count,
                         std::vector<ParameterView> *views) {
    size_t element_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!is_device_memory(tensors[i]->grad)) {
            return -1;
        }
        ParameterView view = {};
        view.value = tensors[i]->data;
        view.gradient = tensors[i]->grad;
        view.count = static_cast<unsigned>(tensor_numel(tensors[i]));
        view.element_offset = static_cast<unsigned>(element_offset);
        element_offset += view.count;
        views->push_back(view);
    }
    return 0;
}

int build_parameter_views(Parameter *const *parameters, size_t count,
                          const float *weight_decays, float beta1, float beta2,
                          std::vector<ParameterView> *views) {
    size_t element_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        Parameter *parameter = parameters[i];
        if (!all_device_memory({parameter->value.data, parameter->value.grad,
                                parameter->m, parameter->v})) {
            return -1;
        }
        ParameterView view = {};
        view.value = parameter->value.data;
        view.gradient = parameter->value.grad;
        view.first_moment = parameter->m;
        view.second_moment = parameter->v;
        view.count = static_cast<unsigned>(tensor_numel(&parameter->value));
        view.element_offset = static_cast<unsigned>(element_offset);
        view.weight_decay = weight_decays ? weight_decays[i] : 0.0f;
        view.correction1 =
            1.0f - powf(beta1, static_cast<float>(parameter->step));
        view.correction2 =
            1.0f - powf(beta2, static_cast<float>(parameter->step));
        element_offset += view.count;
        views->push_back(view);
    }
    return 0;
}

const float *encode_gradient_clip_scale(CudaBackend *backend,
                                        const ParameterViewTable &table,
                                        float max_norm) {
    float *partials = scratch_memory<float>(backend->gradient_norm_partials,
                                            table.view_count);
    float *scale = scratch_memory<float>(backend->gradient_scale, 1);
    squared_norm_views<<<table.view_count, kThreadsPerBlock, 0,
                         backend->stream>>>(table.device_views, partials);
    check_launch("squared_norm_views");
    KernelParameters scale_parameters = {};
    scale_parameters.count = table.view_count;
    scale_parameters.max_norm = max_norm;
    clip_scale<<<1, 1, 0, backend->stream>>>(partials, scale,
                                             scale_parameters);
    check_launch("clip_scale");
    return scale;
}

int device_zero_gradients(void *context, Tensor *const *tensors,
                          size_t tensor_count) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    std::vector<ParameterView> views;
    ParameterViewTable table;
    if (build_gradient_views(tensors, tensor_count, &views) != 0 ||
        upload_view_table(backend, views, &table) != 0) {
        return -1;
    }
    KernelParameters parameters = {};
    parameters.count = static_cast<unsigned>(table.element_count);
    parameters.parameter_index = table.view_count;
    launch_elementwise(backend, "zero_gradients_views", zero_gradients_views,
                       table.element_count, table.device_views, parameters);
    return 0;
}

int device_clip_gradient_norm(void *context, Parameter *const *parameters,
                              size_t parameter_count, float max_norm) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    std::vector<Tensor *> values;
    for (size_t i = 0; i < parameter_count; ++i) {
        values.push_back(&parameters[i]->value);
    }
    std::vector<ParameterView> views;
    ParameterViewTable table;
    if (build_gradient_views(values.data(), values.size(), &views) != 0 ||
        upload_view_table(backend, views, &table) != 0) {
        return -1;
    }
    const float *scale = encode_gradient_clip_scale(backend, table, max_norm);
    KernelParameters scale_parameters = {};
    scale_parameters.count = static_cast<unsigned>(table.element_count);
    scale_parameters.parameter_index = table.view_count;
    launch_elementwise(backend, "scale_gradients_views", scale_gradients_views,
                       table.element_count, table.device_views, scale,
                       scale_parameters);
    return 0;
}

int device_adamw_step_parameters(void *context, Parameter *const *parameters,
                                 const float *weight_decays,
                                 size_t parameter_count, float learning_rate,
                                 float beta1, float beta2, float epsilon,
                                 float max_gradient_norm) {
    CudaBackend *backend = static_cast<CudaBackend *>(context);
    std::vector<ParameterView> views;
    ParameterViewTable table;
    if (build_parameter_views(parameters, parameter_count, weight_decays,
                              beta1, beta2, &views) != 0 ||
        upload_view_table(backend, views, &table) != 0) {
        return -1;
    }
    const float *scale = max_gradient_norm > 0.0f
        ? encode_gradient_clip_scale(backend, table, max_gradient_norm)
        : nullptr;
    KernelParameters adam_parameters = {};
    adam_parameters.count = static_cast<unsigned>(table.element_count);
    adam_parameters.parameter_index = table.view_count;
    adam_parameters.learning_rate = learning_rate;
    adam_parameters.beta1 = beta1;
    adam_parameters.beta2 = beta2;
    adam_parameters.epsilon = epsilon;
    adam_parameters.max_norm = max_gradient_norm;
    launch_elementwise(backend, "adamw_views", adamw_views,
                       table.element_count, table.device_views, scale,
                       adam_parameters);
    return 0;
}

OpsDeviceBackend make_device_backend(CudaBackend *backend) {
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
    device_backend.token_embedding_backward = device_token_embedding_backward;
    device_backend.gather_rows = device_gather_rows;
    device_backend.gather_rows_backward = device_gather_rows_backward;
    device_backend.softmax_cross_entropy = device_softmax_cross_entropy;
    device_backend.clip_gradient_norm = device_clip_gradient_norm;
    device_backend.adamw_step = device_adamw_step;
    device_backend.zero_gradients = device_zero_gradients;
    device_backend.adamw_step_parameters = device_adamw_step_parameters;
    device_backend.synchronize = synchronize_device;
    return device_backend;
}

}

extern "C" int cuda_backend_create(CudaBackend **backend) {
    if (!backend) {
        return -1;
    }
    *backend = nullptr;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        cudaGetLastError();
        return 1;
    }
    const int device = 0;
    cudaDeviceProp properties = {};
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        !properties.managedMemory) {
        cudaGetLastError();
        return 1;
    }
    CudaBackend *result = new CudaBackend();
    result->device = device;
    result->max_shared_memory_per_block = properties.sharedMemPerBlock;
    if (cudaStreamCreate(&result->stream) != cudaSuccess) {
        delete result;
        return -1;
    }
    if (cublasCreate(&result->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(result->blas, result->stream) !=
            CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(result->stream);
        delete result;
        return -1;
    }
    *backend = result;
    return 0;
}

extern "C" void cuda_backend_free(CudaBackend *backend) {
    if (!backend) {
        return;
    }
    cuda_backend_disable_device_execution(backend);
    cudaStreamSynchronize(backend->stream);
    release_scratch(backend->layer_norm_statistics);
    release_scratch(backend->row_losses);
    release_scratch(backend->gradient_norm_partials);
    release_scratch(backend->gradient_scale);
    release_scratch(backend->parameter_views);
    cublasDestroy(backend->blas);
    cudaStreamDestroy(backend->stream);
    delete backend;
}

extern "C" int cuda_backend_available(const CudaBackend *backend) {
    return backend && backend->stream && backend->blas ? 1 : 0;
}

extern "C" int cuda_backend_enable_device_execution(CudaBackend *backend) {
    if (!cuda_backend_available(backend)) {
        return -1;
    }
    backend->device_backend = make_device_backend(backend);
    ops_set_device_backend(&backend->device_backend);
    tensor_set_allocator(allocate_device_memory, release_device_memory,
                         backend);
    backend->device_execution_enabled = 1;
    return 0;
}

extern "C" void cuda_backend_disable_device_execution(CudaBackend *backend) {
    if (!backend || !backend->device_execution_enabled) {
        return;
    }
    ops_synchronize();
    ops_set_device_backend(nullptr);
    tensor_set_allocator(nullptr, nullptr, nullptr);
    backend->device_execution_enabled = 0;
}

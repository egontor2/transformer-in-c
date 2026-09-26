#include <metal_stdlib>
using namespace metal;

struct KernelParameters {
    uint count;
    uint rows;
    uint cols;
    uint batch;
    uint sequence_length;
    uint d_model;
    uint heads;
    uint channels;
    uint height;
    uint width;
    uint patch_size;
    uint parameter_index;
    uint has_class_weights;
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

constant float gelu_cubic_coefficient = 0.044715f;
constant float gelu_tanh_scale = 0.7978845608028654f;

kernel void accumulate_gradient(device float *destination [[buffer(0)]],
                                device const float *source [[buffer(1)]],
                                constant KernelParameters &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        destination[i] += source[i];
    }
}

kernel void residual_forward(device const float *left [[buffer(0)]],
                             device const float *right [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant KernelParameters &p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        output[i] = left[i] + right[i];
    }
}

kernel void residual_backward(device float *left_grad [[buffer(0)]],
                              device float *right_grad [[buffer(1)]],
                              device const float *output_grad [[buffer(2)]],
                              constant KernelParameters &p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        left_grad[i] += output_grad[i];
        right_grad[i] += output_grad[i];
    }
}

kernel void bias_add_forward(device const float *input [[buffer(0)]],
                             device const float *bias [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant KernelParameters &p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        output[i] = input[i] + bias[i % p.cols];
    }
}

kernel void gelu_forward(device const float *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         constant KernelParameters &p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        const float x = input[i];
        output[i] = 0.5f * x *
            (1.0f + precise::tanh(gelu_tanh_scale *
                                  (x + gelu_cubic_coefficient * x * x * x)));
    }
}

kernel void gelu_backward(device const float *input [[buffer(0)]],
                          device float *input_grad [[buffer(1)]],
                          device const float *output_grad [[buffer(2)]],
                          constant KernelParameters &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const float x = input[i];
    const float tanh_inner = precise::tanh(
        gelu_tanh_scale * (x + gelu_cubic_coefficient * x * x * x));
    const float derivative = 0.5f * (1.0f + tanh_inner) +
        0.5f * x * (1.0f - tanh_inner * tanh_inner) * gelu_tanh_scale *
        (1.0f + 3.0f * gelu_cubic_coefficient * x * x);
    input_grad[i] += output_grad[i] * derivative;
}

static uint head_major_index(uint row, uint feature, constant KernelParameters &p) {
    const uint head_dimension = p.d_model / p.heads;
    const uint sample = row / p.sequence_length;
    const uint position = row % p.sequence_length;
    const uint head = feature / head_dimension;
    const uint dimension = feature % head_dimension;
    return ((sample * p.heads + head) * p.sequence_length + position) *
           head_dimension + dimension;
}

kernel void split_heads_forward(device const float *projected [[buffer(0)]],
                                device float *query [[buffer(1)]],
                                device float *key [[buffer(2)]],
                                device float *value [[buffer(3)]],
                                constant KernelParameters &p [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint row = i / p.d_model;
    const uint feature = i % p.d_model;
    const uint target = head_major_index(row, feature, p);
    device const float *source = projected + row * 3 * p.d_model;
    query[target] = source[feature];
    key[target] = source[p.d_model + feature];
    value[target] = source[2 * p.d_model + feature];
}

kernel void split_heads_backward(device float *projected_grad [[buffer(0)]],
                                 device const float *query_grad [[buffer(1)]],
                                 device const float *key_grad [[buffer(2)]],
                                 device const float *value_grad [[buffer(3)]],
                                 constant KernelParameters &p [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint row = i / p.d_model;
    const uint feature = i % p.d_model;
    const uint source = head_major_index(row, feature, p);
    device float *destination = projected_grad + row * 3 * p.d_model;
    destination[feature] += query_grad[source];
    destination[p.d_model + feature] += key_grad[source];
    destination[2 * p.d_model + feature] += value_grad[source];
}

kernel void merge_heads_forward(device const float *attended [[buffer(0)]],
                                device float *merged [[buffer(1)]],
                                constant KernelParameters &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        merged[i] = attended[head_major_index(i / p.d_model, i % p.d_model, p)];
    }
}

kernel void merge_heads_backward(device float *attended_grad [[buffer(0)]],
                                 device const float *merged_grad [[buffer(1)]],
                                 constant KernelParameters &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        attended_grad[head_major_index(i / p.d_model, i % p.d_model, p)] +=
            merged_grad[i];
    }
}

kernel void attention_forward(device const float *query [[buffer(0)]],
                              device const float *key [[buffer(1)]],
                              device const float *value [[buffer(2)]],
                              device float *probabilities [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              constant KernelParameters &p [[buffer(5)]],
                              uint query_row [[thread_position_in_grid]]) {
    if (query_row >= p.rows) {
        return;
    }
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_start = (query_row / length) * length;
    device const float *query_values = query + query_row * head_dimension;
    device float *probability_row = probabilities + query_row * length;
    float maximum = -INFINITY;
    for (uint key_index = 0; key_index < length; ++key_index) {
        device const float *key_values =
            key + (group_start + key_index) * head_dimension;
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            score += query_values[dimension] * key_values[dimension];
        }
        score *= p.scale;
        probability_row[key_index] = score;
        maximum = max(maximum, score);
    }
    float denominator = 0.0f;
    for (uint key_index = 0; key_index < length; ++key_index) {
        probability_row[key_index] =
            precise::exp(probability_row[key_index] - maximum);
        denominator += probability_row[key_index];
    }
    for (uint key_index = 0; key_index < length; ++key_index) {
        probability_row[key_index] /= denominator;
    }
    for (uint dimension = 0; dimension < head_dimension; ++dimension) {
        float result = 0.0f;
        for (uint key_index = 0; key_index < length; ++key_index) {
            result += probability_row[key_index] *
                      value[(group_start + key_index) * head_dimension + dimension];
        }
        output[query_row * head_dimension + dimension] = result;
    }
}

kernel void attention_backward_query(device const float *key [[buffer(0)]],
                                     device const float *value [[buffer(1)]],
                                     device const float *probabilities [[buffer(2)]],
                                     device const float *output_grad [[buffer(3)]],
                                     device float *score_grad [[buffer(4)]],
                                     device float *query_grad [[buffer(5)]],
                                     constant KernelParameters &p [[buffer(6)]],
                                     uint query_row [[thread_position_in_grid]]) {
    if (query_row >= p.rows) {
        return;
    }
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_start = (query_row / length) * length;
    device const float *upstream = output_grad + query_row * head_dimension;
    device const float *probability_row = probabilities + query_row * length;
    device float *score_row = score_grad + query_row * length;
    float probability_dot_gradient = 0.0f;
    for (uint key_index = 0; key_index < length; ++key_index) {
        device const float *value_values =
            value + (group_start + key_index) * head_dimension;
        float probability_gradient = 0.0f;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            probability_gradient += upstream[dimension] * value_values[dimension];
        }
        score_row[key_index] = probability_gradient;
        probability_dot_gradient += probability_row[key_index] * probability_gradient;
    }
    for (uint key_index = 0; key_index < length; ++key_index) {
        const float score_gradient = p.scale * probability_row[key_index] *
            (score_row[key_index] - probability_dot_gradient);
        score_row[key_index] = score_gradient;
        device const float *key_values =
            key + (group_start + key_index) * head_dimension;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            query_grad[query_row * head_dimension + dimension] +=
                score_gradient * key_values[dimension];
        }
    }
}

kernel void attention_backward_key_value(
    device const float *query [[buffer(0)]],
    device const float *probabilities [[buffer(1)]],
    device const float *output_grad [[buffer(2)]],
    device const float *score_grad [[buffer(3)]],
    device float *key_grad [[buffer(4)]],
    device float *value_grad [[buffer(5)]],
    constant KernelParameters &p [[buffer(6)]],
    uint key_row [[thread_position_in_grid]]) {
    if (key_row >= p.rows) {
        return;
    }
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_start = (key_row / length) * length;
    const uint key_index = key_row - group_start;
    for (uint query_index = 0; query_index < length; ++query_index) {
        const uint query_row = group_start + query_index;
        const float score_gradient = score_grad[query_row * length + key_index];
        const float probability = probabilities[query_row * length + key_index];
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            key_grad[key_row * head_dimension + dimension] +=
                score_gradient * query[query_row * head_dimension + dimension];
            value_grad[key_row * head_dimension + dimension] +=
                probability * output_grad[query_row * head_dimension + dimension];
        }
    }
}

kernel void zero_gradient(device float *gradient [[buffer(0)]],
                          constant KernelParameters &p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        gradient[i] = 0.0f;
    }
}

kernel void extract_patches(device const float *images [[buffer(0)]],
                            device float *patches [[buffer(1)]],
                            constant KernelParameters &p [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint patch_row = i / p.cols;
    const uint element = i % p.cols;
    const uint patches_per_row = p.width / p.patch_size;
    const uint patches_per_image = (p.height / p.patch_size) * patches_per_row;
    const uint sample = patch_row / patches_per_image;
    const uint patch_index = patch_row % patches_per_image;
    const uint channel = element / (p.patch_size * p.patch_size);
    const uint y = (patch_index / patches_per_row) * p.patch_size +
                   (element / p.patch_size) % p.patch_size;
    const uint x = (patch_index % patches_per_row) * p.patch_size +
                   element % p.patch_size;
    patches[i] = images[((sample * p.channels + channel) * p.height + y) *
                        p.width + x];
}

kernel void token_embedding_forward(device const float *patch_tokens [[buffer(0)]],
                                    device const float *cls_token [[buffer(1)]],
                                    device const float *positional [[buffer(2)]],
                                    device float *tokens [[buffer(3)]],
                                    constant KernelParameters &p [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint dimension = i % p.d_model;
    const uint position = (i / p.d_model) % p.sequence_length;
    const uint sample = i / (p.d_model * p.sequence_length);
    const uint patch_count = p.sequence_length - 1;
    const float token = position == 0
        ? cls_token[dimension]
        : patch_tokens[(sample * patch_count + position - 1) * p.d_model +
                       dimension];
    tokens[i] = token + positional[position * p.d_model + dimension];
}

kernel void token_embedding_backward_patches(
    device float *patch_grad [[buffer(0)]],
    device const float *tokens_grad [[buffer(1)]],
    constant KernelParameters &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint patch_count = p.sequence_length - 1;
    const uint dimension = i % p.d_model;
    const uint patch = (i / p.d_model) % patch_count;
    const uint sample = i / (p.d_model * patch_count);
    patch_grad[i] += tokens_grad[(sample * p.sequence_length + patch + 1) *
                                 p.d_model + dimension];
}

kernel void token_embedding_backward_parameters(
    device float *cls_grad [[buffer(0)]],
    device float *positional_grad [[buffer(1)]],
    device const float *tokens_grad [[buffer(2)]],
    constant KernelParameters &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const uint dimension = i % p.d_model;
    const uint position = i / p.d_model;
    float sum = 0.0f;
    for (uint sample = 0; sample < p.batch; ++sample) {
        sum += tokens_grad[(sample * p.sequence_length + position) * p.d_model +
                           dimension];
    }
    positional_grad[i] += sum;
    if (position == 0) {
        cls_grad[dimension] += sum;
    }
}

kernel void gather_rows_forward(device const float *source [[buffer(0)]],
                                device float *destination [[buffer(1)]],
                                constant KernelParameters &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        destination[i] =
            source[(i / p.cols) * p.sequence_length * p.cols + i % p.cols];
    }
}

kernel void gather_rows_backward(device float *source_grad [[buffer(0)]],
                                 device const float *destination_grad [[buffer(1)]],
                                 constant KernelParameters &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i < p.count) {
        source_grad[(i / p.cols) * p.sequence_length * p.cols + i % p.cols] +=
            destination_grad[i];
    }
}

kernel void softmax_cross_entropy_rows(device const float *logits [[buffer(0)]],
                                       device const float *targets [[buffer(1)]],
                                       device const float *class_weights [[buffer(2)]],
                                       device float *logits_grad [[buffer(3)]],
                                       device float *row_losses [[buffer(4)]],
                                       constant KernelParameters &p [[buffer(5)]],
                                       uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) {
        return;
    }
    device const float *row_logits = logits + row * p.cols;
    const uint target = uint(targets[row]);
    float maximum = -INFINITY;
    for (uint col = 0; col < p.cols; ++col) {
        maximum = max(maximum, row_logits[col]);
    }
    float denominator = 0.0f;
    for (uint col = 0; col < p.cols; ++col) {
        denominator += precise::exp(row_logits[col] - maximum);
    }
    const float log_denominator = maximum + precise::log(denominator);
    const float weight = p.has_class_weights ? class_weights[target] : 1.0f;
    const float off_target_probability = p.label_smoothing / float(p.cols);
    const float on_target_probability =
        1.0f - p.label_smoothing + off_target_probability;
    float loss = 0.0f;
    for (uint col = 0; col < p.cols; ++col) {
        const float target_probability =
            col == target ? on_target_probability : off_target_probability;
        const float probability =
            precise::exp(row_logits[col] - log_denominator);
        loss += weight * target_probability * (log_denominator - row_logits[col]);
        logits_grad[row * p.cols + col] +=
            weight * (probability - target_probability) / float(p.rows);
    }
    row_losses[row] = loss;
}

kernel void mean_loss(device const float *row_losses [[buffer(0)]],
                      device float *loss [[buffer(1)]],
                      constant KernelParameters &p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i != 0) {
        return;
    }
    float sum = 0.0f;
    for (uint row = 0; row < p.rows; ++row) {
        sum += row_losses[row];
    }
    loss[0] = sum / float(p.rows);
}

kernel void clip_scale(device const float *partials [[buffer(0)]],
                       device float *scale [[buffer(1)]],
                       constant KernelParameters &p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i != 0) {
        return;
    }
    float squared_norm = 0.0f;
    for (uint index = 0; index < p.count; ++index) {
        squared_norm += partials[index];
    }
    const float norm = precise::sqrt(squared_norm);
    scale[0] = norm > p.max_norm && norm > 0.0f ? p.max_norm / norm : 1.0f;
}

kernel void adamw_step(device float *value [[buffer(0)]],
                       device float *gradient [[buffer(1)]],
                       device float *first_moment [[buffer(2)]],
                       device float *second_moment [[buffer(3)]],
                       constant KernelParameters &p [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    const float g = gradient[i];
    first_moment[i] = p.beta1 * first_moment[i] + (1.0f - p.beta1) * g;
    second_moment[i] = p.beta2 * second_moment[i] + (1.0f - p.beta2) * g * g;
    const float estimate = (first_moment[i] / p.correction1) /
        (precise::sqrt(second_moment[i] / p.correction2) + p.epsilon);
    value[i] -= p.learning_rate * (estimate + p.weight_decay * value[i]);
    gradient[i] = 0.0f;
}

kernel void column_sums_tiled(device const float *values [[buffer(0)]],
                              device float *column_sums [[buffer(1)]],
                              constant KernelParameters &p [[buffer(2)]],
                              threadgroup float *partial_sums [[threadgroup(0)]],
                              uint2 tile [[threadgroup_position_in_grid]],
                              uint2 lane [[thread_position_in_threadgroup]],
                              uint2 lanes [[threads_per_threadgroup]]) {
    const uint col = tile.x * lanes.x + lane.x;
    float sum = 0.0f;
    if (col < p.cols) {
        for (uint row = lane.y; row < p.rows; row += lanes.y) {
            sum += values[row * p.cols + col];
        }
    }
    partial_sums[lane.y * lanes.x + lane.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes.y / 2; stride > 0; stride /= 2) {
        if (lane.y < stride) {
            partial_sums[lane.y * lanes.x + lane.x] +=
                partial_sums[(lane.y + stride) * lanes.x + lane.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane.y == 0 && col < p.cols) {
        column_sums[col] += partial_sums[lane.x];
    }
}

static void simd_row_statistics(device const float *row_values, uint cols,
                                float epsilon, uint lane, thread float &mean,
                                thread float &inverse_std) {
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        sum += row_values[col];
    }
    mean = simd_sum(sum) / float(cols);
    float variance = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const float centered = row_values[col] - mean;
        variance += centered * centered;
    }
    inverse_std = 1.0f / precise::sqrt(simd_sum(variance) / float(cols) + epsilon);
}

kernel void layer_norm_forward_simd(device const float *input [[buffer(0)]],
                                    device const float *gamma [[buffer(1)]],
                                    device const float *beta [[buffer(2)]],
                                    device float *output [[buffer(3)]],
                                    constant KernelParameters &p [[buffer(4)]],
                                    uint thread_id [[thread_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_id / 32;
    if (row >= p.rows) {
        return;
    }
    device const float *row_values = input + row * p.cols;
    float mean;
    float inverse_std;
    simd_row_statistics(row_values, p.cols, p.epsilon, lane, mean, inverse_std);
    for (uint col = lane; col < p.cols; col += 32) {
        output[row * p.cols + col] =
            (row_values[col] - mean) * inverse_std * gamma[col] + beta[col];
    }
}

kernel void layer_norm_backward_rows_simd(device const float *input [[buffer(0)]],
                                          device const float *gamma [[buffer(1)]],
                                          device const float *output_grad [[buffer(2)]],
                                          device float *input_grad [[buffer(3)]],
                                          device float *statistics [[buffer(4)]],
                                          constant KernelParameters &p [[buffer(5)]],
                                          uint thread_id [[thread_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint row = thread_id / 32;
    if (row >= p.rows) {
        return;
    }
    device const float *row_values = input + row * p.cols;
    device const float *upstream = output_grad + row * p.cols;
    float mean;
    float inverse_std;
    simd_row_statistics(row_values, p.cols, p.epsilon, lane, mean, inverse_std);
    if (lane == 0) {
        statistics[2 * row] = mean;
        statistics[2 * row + 1] = inverse_std;
    }
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (uint col = lane; col < p.cols; col += 32) {
        const float normalized = (row_values[col] - mean) * inverse_std;
        const float dxhat = upstream[col] * gamma[col];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * normalized;
    }
    const float cols = float(p.cols);
    const float mean_dxhat = simd_sum(sum_dxhat) / cols;
    const float mean_dxhat_xhat = simd_sum(sum_dxhat_xhat) / cols;
    for (uint col = lane; col < p.cols; col += 32) {
        const float normalized = (row_values[col] - mean) * inverse_std;
        const float dxhat = upstream[col] * gamma[col];
        input_grad[row * p.cols + col] +=
            inverse_std * (dxhat - mean_dxhat - normalized * mean_dxhat_xhat);
    }
}

kernel void layer_norm_parameter_gradients_tiled(
    device const float *input [[buffer(0)]],
    device const float *output_grad [[buffer(1)]],
    device const float *statistics [[buffer(2)]],
    device float *gamma_grad [[buffer(3)]],
    device float *beta_grad [[buffer(4)]],
    constant KernelParameters &p [[buffer(5)]],
    threadgroup float *partial_sums [[threadgroup(0)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint2 lanes [[threads_per_threadgroup]]) {
    const uint col = tile.x * lanes.x + lane.x;
    const uint tile_size = lanes.x * lanes.y;
    const uint slot = lane.y * lanes.x + lane.x;
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    if (col < p.cols) {
        for (uint row = lane.y; row < p.rows; row += lanes.y) {
            const float upstream = output_grad[row * p.cols + col];
            const float normalized = (input[row * p.cols + col] -
                                      statistics[2 * row]) *
                                     statistics[2 * row + 1];
            gamma_sum += upstream * normalized;
            beta_sum += upstream;
        }
    }
    partial_sums[slot] = gamma_sum;
    partial_sums[tile_size + slot] = beta_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes.y / 2; stride > 0; stride /= 2) {
        if (lane.y < stride) {
            partial_sums[slot] += partial_sums[slot + stride * lanes.x];
            partial_sums[tile_size + slot] +=
                partial_sums[tile_size + slot + stride * lanes.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane.y == 0 && col < p.cols) {
        gamma_grad[col] += partial_sums[lane.x];
        beta_grad[col] += partial_sums[tile_size + lane.x];
    }
}

kernel void attention_forward_shared(device const float *query [[buffer(0)]],
                                     device const float *key [[buffer(1)]],
                                     device const float *value [[buffer(2)]],
                                     device float *probabilities [[buffer(3)]],
                                     device float *output [[buffer(4)]],
                                     constant KernelParameters &p [[buffer(5)]],
                                     threadgroup float *shared [[threadgroup(0)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_threadgroup]],
                                     uint lanes [[threads_per_threadgroup]],
                                     uint simd_lane [[thread_index_in_simdgroup]],
                                     uint simd_group [[simdgroup_index_in_threadgroup]],
                                     uint simd_groups [[simdgroups_per_threadgroup]]) {
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_elements = length * head_dimension;
    const uint score_count = length * length;
    threadgroup float *shared_queries = shared;
    threadgroup float *shared_keys = shared_queries + group_elements;
    threadgroup float *shared_values = shared_keys + group_elements;
    threadgroup float *shared_scores = shared_values + group_elements;
    const uint group_offset = group * group_elements;
    for (uint i = lane; i < group_elements; i += lanes) {
        shared_queries[i] = query[group_offset + i];
        shared_keys[i] = key[group_offset + i];
        shared_values[i] = value[group_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pair = lane; pair < score_count; pair += lanes) {
        const uint query_index = pair / length;
        const uint key_index = pair % length;
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            score += shared_queries[query_index * head_dimension + dimension] *
                     shared_keys[key_index * head_dimension + dimension];
        }
        shared_scores[pair] = score * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query_index = simd_group; query_index < length;
         query_index += simd_groups) {
        threadgroup float *score_row = shared_scores + query_index * length;
        float maximum = -INFINITY;
        for (uint key_index = simd_lane; key_index < length; key_index += 32) {
            maximum = max(maximum, score_row[key_index]);
        }
        maximum = simd_max(maximum);
        float denominator = 0.0f;
        for (uint key_index = simd_lane; key_index < length; key_index += 32) {
            score_row[key_index] = precise::exp(score_row[key_index] - maximum);
            denominator += score_row[key_index];
        }
        denominator = simd_sum(denominator);
        device float *probability_output =
            probabilities + (group * length + query_index) * length;
        for (uint key_index = simd_lane; key_index < length; key_index += 32) {
            score_row[key_index] /= denominator;
            probability_output[key_index] = score_row[key_index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint element = lane; element < group_elements; element += lanes) {
        const uint query_index = element / head_dimension;
        const uint dimension = element % head_dimension;
        float result = 0.0f;
        for (uint key_index = 0; key_index < length; ++key_index) {
            result += shared_scores[query_index * length + key_index] *
                      shared_values[key_index * head_dimension + dimension];
        }
        output[group_offset + element] = result;
    }
}

kernel void attention_backward_shared(device const float *query [[buffer(0)]],
                                      device const float *key [[buffer(1)]],
                                      device const float *value [[buffer(2)]],
                                      device const float *probabilities [[buffer(3)]],
                                      device const float *output [[buffer(4)]],
                                      device const float *output_grad [[buffer(5)]],
                                      device float *query_grad [[buffer(6)]],
                                      device float *key_grad [[buffer(7)]],
                                      device float *value_grad [[buffer(8)]],
                                      constant KernelParameters &p [[buffer(9)]],
                                      threadgroup float *shared [[threadgroup(0)]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]],
                                      uint lanes [[threads_per_threadgroup]]) {
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_elements = length * head_dimension;
    const uint score_count = length * length;
    threadgroup float *shared_queries = shared;
    threadgroup float *shared_keys = shared_queries + group_elements;
    threadgroup float *shared_values = shared_keys + group_elements;
    threadgroup float *shared_upstream = shared_values + group_elements;
    threadgroup float *shared_scores = shared_upstream + group_elements;
    threadgroup float *upstream_dot_output = shared_scores + score_count;
    const uint group_offset = group * group_elements;
    for (uint i = lane; i < group_elements; i += lanes) {
        shared_queries[i] = query[group_offset + i];
        shared_keys[i] = key[group_offset + i];
        shared_values[i] = value[group_offset + i];
        shared_upstream[i] = output_grad[group_offset + i];
    }
    for (uint i = lane; i < score_count; i += lanes) {
        shared_scores[i] = probabilities[group * score_count + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query_index = lane; query_index < length; query_index += lanes) {
        float dot = 0.0f;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            const uint element = query_index * head_dimension + dimension;
            dot += shared_upstream[element] * output[group_offset + element];
        }
        upstream_dot_output[query_index] = dot;
    }
    for (uint element = lane; element < group_elements; element += lanes) {
        const uint key_index = element / head_dimension;
        const uint dimension = element % head_dimension;
        float gradient = 0.0f;
        for (uint query_index = 0; query_index < length; ++query_index) {
            gradient += shared_scores[query_index * length + key_index] *
                        shared_upstream[query_index * head_dimension + dimension];
        }
        value_grad[group_offset + element] += gradient;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pair = lane; pair < score_count; pair += lanes) {
        const uint query_index = pair / length;
        const uint key_index = pair % length;
        float probability_gradient = 0.0f;
        for (uint dimension = 0; dimension < head_dimension; ++dimension) {
            probability_gradient +=
                shared_upstream[query_index * head_dimension + dimension] *
                shared_values[key_index * head_dimension + dimension];
        }
        shared_scores[pair] = p.scale * shared_scores[pair] *
            (probability_gradient - upstream_dot_output[query_index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint work = lane; work < 2 * group_elements; work += lanes) {
        const uint element = work % group_elements;
        const uint row = element / head_dimension;
        const uint dimension = element % head_dimension;
        float gradient = 0.0f;
        if (work < group_elements) {
            for (uint key_index = 0; key_index < length; ++key_index) {
                gradient += shared_scores[row * length + key_index] *
                            shared_keys[key_index * head_dimension + dimension];
            }
            query_grad[group_offset + element] += gradient;
        } else {
            for (uint query_index = 0; query_index < length; ++query_index) {
                gradient += shared_scores[query_index * length + row] *
                            shared_queries[query_index * head_dimension + dimension];
            }
            key_grad[group_offset + element] += gradient;
        }
    }
}

struct ParameterView {
    device float *value;
    device float *gradient;
    device float *first_moment;
    device float *second_moment;
    uint count;
    uint element_offset;
    float weight_decay;
    float correction1;
    float correction2;
    uint padding;
};

static uint find_view(device const ParameterView *views, uint view_count,
                      uint element) {
    uint low = 0;
    uint high = view_count - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (views[middle].element_offset <= element) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

kernel void zero_gradients_views(device const ParameterView *views [[buffer(0)]],
                                 constant KernelParameters &p [[buffer(1)]],
                                 uint element [[thread_position_in_grid]]) {
    if (element >= p.count) {
        return;
    }
    const uint view = find_view(views, p.parameter_index, element);
    views[view].gradient[element - views[view].element_offset] = 0.0f;
}

kernel void squared_norm_views(device const ParameterView *views [[buffer(0)]],
                               device float *partials [[buffer(1)]],
                               constant KernelParameters &p [[buffer(2)]],
                               uint view [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]],
                               uint lanes [[threads_per_threadgroup]]) {
    threadgroup float shared_sums[256];
    device const float *gradient = views[view].gradient;
    float sum = 0.0f;
    for (uint i = lane; i < views[view].count; i += lanes) {
        sum += gradient[i] * gradient[i];
    }
    shared_sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partials[view] = shared_sums[0];
    }
}

kernel void scale_gradients_views(device const ParameterView *views [[buffer(0)]],
                                  device const float *scale [[buffer(1)]],
                                  constant KernelParameters &p [[buffer(2)]],
                                  uint element [[thread_position_in_grid]]) {
    if (element >= p.count) {
        return;
    }
    const uint view = find_view(views, p.parameter_index, element);
    views[view].gradient[element - views[view].element_offset] *= scale[0];
}

kernel void adamw_views(device const ParameterView *views [[buffer(0)]],
                        device const float *gradient_scale [[buffer(1)]],
                        constant KernelParameters &p [[buffer(2)]],
                        uint element [[thread_position_in_grid]]) {
    if (element >= p.count) {
        return;
    }
    const uint view_index = find_view(views, p.parameter_index, element);
    device const ParameterView &view = views[view_index];
    const uint i = element - view.element_offset;
    const float g = p.max_norm > 0.0f ? view.gradient[i] * gradient_scale[0]
                                      : view.gradient[i];
    const float first = p.beta1 * view.first_moment[i] + (1.0f - p.beta1) * g;
    const float second = p.beta2 * view.second_moment[i] +
                         (1.0f - p.beta2) * g * g;
    view.first_moment[i] = first;
    view.second_moment[i] = second;
    const float estimate = (first / view.correction1) /
        (precise::sqrt(second / view.correction2) + p.epsilon);
    view.value[i] -= p.learning_rate * (estimate + view.weight_decay * view.value[i]);
    view.gradient[i] = 0.0f;
}

constant uint flash_block_items = 32;
constant uint flash_lanes_per_item = 8;
constant uint flash_max_head_dimension = 64;

static float2 combine_softmax_statistics(float2 first, float2 second) {
    if (second.x == -INFINITY) {
        return first;
    }
    if (first.x == -INFINITY) {
        return second;
    }
    const float maximum = max(first.x, second.x);
    return float2(maximum, first.y * precise::exp(first.x - maximum) +
                               second.y * precise::exp(second.x - maximum));
}

static float2 reduce_softmax_statistics_across_lanes(float2 statistics) {
    for (ushort offset = 4; offset > 0; offset /= 2) {
        const float2 other = float2(simd_shuffle_xor(statistics.x, offset),
                                    simd_shuffle_xor(statistics.y, offset));
        statistics = combine_softmax_statistics(statistics, other);
    }
    return statistics;
}

static float reduce_sum_across_lanes(float value) {
    for (ushort offset = 4; offset > 0; offset /= 2) {
        value += simd_shuffle_xor(value, offset);
    }
    return value;
}

static void load_tile(device const float *source, threadgroup float *tile,
                      uint first_row, uint row_count, uint head_dimension,
                      uint thread_id) {
    const uint threads = flash_block_items * flash_lanes_per_item;
    for (uint i = thread_id; i < flash_block_items * head_dimension;
         i += threads) {
        const uint row = i / head_dimension;
        tile[i] = row < row_count ? source[(first_row + row) * head_dimension +
                                           i % head_dimension]
                                  : 0.0f;
    }
}

kernel void attention_forward_flash(device const float *query [[buffer(0)]],
                                    device const float *key [[buffer(1)]],
                                    device const float *value [[buffer(2)]],
                                    device float *probabilities [[buffer(3)]],
                                    device float *output [[buffer(4)]],
                                    constant KernelParameters &p [[buffer(5)]],
                                    uint2 block [[threadgroup_position_in_grid]],
                                    uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup float key_tile[flash_block_items * flash_max_head_dimension];
    threadgroup float value_tile[flash_block_items * flash_max_head_dimension];
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group = block.y;
    const uint group_start = group * length;
    const uint query_index = block.x * flash_block_items +
                             thread_id / flash_lanes_per_item;
    const uint lane = thread_id % flash_lanes_per_item;
    const bool active = query_index < length;
    float query_values[flash_max_head_dimension];
    for (uint d = 0; d < head_dimension; ++d) {
        query_values[d] = active
            ? query[(group_start + query_index) * head_dimension + d]
            : 0.0f;
    }

    float2 statistics = float2(-INFINITY, 0.0f);
    for (uint tile_start = 0; tile_start < length;
         tile_start += flash_block_items) {
        const uint tile_rows = min(flash_block_items, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = lane; active && k < tile_rows; k += flash_lanes_per_item) {
            float score = 0.0f;
            for (uint d = 0; d < head_dimension; ++d) {
                score += query_values[d] * key_tile[k * head_dimension + d];
            }
            statistics = combine_softmax_statistics(
                statistics, float2(score * p.scale, 1.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    statistics = reduce_softmax_statistics_across_lanes(statistics);

    float output_values[flash_max_head_dimension];
    for (uint d = 0; d < head_dimension; ++d) {
        output_values[d] = 0.0f;
    }
    device float *probability_row =
        probabilities + (group_start + query_index) * length;
    for (uint tile_start = 0; tile_start < length;
         tile_start += flash_block_items) {
        const uint tile_rows = min(flash_block_items, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        load_tile(value, value_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = lane; active && k < tile_rows; k += flash_lanes_per_item) {
            float score = 0.0f;
            for (uint d = 0; d < head_dimension; ++d) {
                score += query_values[d] * key_tile[k * head_dimension + d];
            }
            const float probability =
                precise::exp(score * p.scale - statistics.x) / statistics.y;
            probability_row[tile_start + k] = probability;
            for (uint d = 0; d < head_dimension; ++d) {
                output_values[d] += probability * value_tile[k * head_dimension + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < head_dimension; ++d) {
        const float total = reduce_sum_across_lanes(output_values[d]);
        if (active && lane == 0) {
            output[(group_start + query_index) * head_dimension + d] = total;
        }
    }
}

kernel void attention_backward_flash_queries(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device const float *probabilities [[buffer(2)]],
    device const float *output [[buffer(3)]],
    device const float *output_grad [[buffer(4)]],
    device float *score_grad [[buffer(5)]],
    device float *query_grad [[buffer(6)]],
    constant KernelParameters &p [[buffer(7)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup float key_tile[flash_block_items * flash_max_head_dimension];
    threadgroup float value_tile[flash_block_items * flash_max_head_dimension];
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_start = block.y * length;
    const uint query_index = block.x * flash_block_items +
                             thread_id / flash_lanes_per_item;
    const uint lane = thread_id % flash_lanes_per_item;
    const bool active = query_index < length;
    const uint query_row = group_start + query_index;
    float upstream[flash_max_head_dimension];
    float upstream_dot_output = 0.0f;
    for (uint d = 0; d < head_dimension; ++d) {
        upstream[d] = active ? output_grad[query_row * head_dimension + d] : 0.0f;
        upstream_dot_output += active
            ? upstream[d] * output[query_row * head_dimension + d]
            : 0.0f;
    }
    float query_gradient[flash_max_head_dimension];
    for (uint d = 0; d < head_dimension; ++d) {
        query_gradient[d] = 0.0f;
    }
    for (uint tile_start = 0; tile_start < length;
         tile_start += flash_block_items) {
        const uint tile_rows = min(flash_block_items, length - tile_start);
        load_tile(key, key_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        load_tile(value, value_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = lane; active && k < tile_rows; k += flash_lanes_per_item) {
            float probability_gradient = 0.0f;
            for (uint d = 0; d < head_dimension; ++d) {
                probability_gradient += upstream[d] * value_tile[k * head_dimension + d];
            }
            const uint score_index = query_row * length + tile_start + k;
            const float score_gradient = p.scale * probabilities[score_index] *
                (probability_gradient - upstream_dot_output);
            score_grad[score_index] = score_gradient;
            for (uint d = 0; d < head_dimension; ++d) {
                query_gradient[d] += score_gradient * key_tile[k * head_dimension + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < head_dimension; ++d) {
        const float total = reduce_sum_across_lanes(query_gradient[d]);
        if (active && lane == 0) {
            query_grad[query_row * head_dimension + d] += total;
        }
    }
}

kernel void attention_backward_flash_keys(
    device const float *query [[buffer(0)]],
    device const float *probabilities [[buffer(1)]],
    device const float *output_grad [[buffer(2)]],
    device const float *score_grad [[buffer(3)]],
    device float *key_grad [[buffer(4)]],
    device float *value_grad [[buffer(5)]],
    constant KernelParameters &p [[buffer(6)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup float query_tile[flash_block_items * flash_max_head_dimension];
    threadgroup float upstream_tile[flash_block_items * flash_max_head_dimension];
    const uint length = p.sequence_length;
    const uint head_dimension = p.cols;
    const uint group_start = block.y * length;
    const uint key_index = block.x * flash_block_items +
                           thread_id / flash_lanes_per_item;
    const uint lane = thread_id % flash_lanes_per_item;
    const bool active = key_index < length;
    float key_gradient[flash_max_head_dimension];
    float value_gradient[flash_max_head_dimension];
    for (uint d = 0; d < head_dimension; ++d) {
        key_gradient[d] = 0.0f;
        value_gradient[d] = 0.0f;
    }
    for (uint tile_start = 0; tile_start < length;
         tile_start += flash_block_items) {
        const uint tile_rows = min(flash_block_items, length - tile_start);
        load_tile(query, query_tile, group_start + tile_start, tile_rows,
                  head_dimension, thread_id);
        load_tile(output_grad, upstream_tile, group_start + tile_start,
                  tile_rows, head_dimension, thread_id);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = lane; active && q < tile_rows; q += flash_lanes_per_item) {
            const uint score_index =
                (group_start + tile_start + q) * length + key_index;
            const float probability = probabilities[score_index];
            const float score_gradient = score_grad[score_index];
            for (uint d = 0; d < head_dimension; ++d) {
                key_gradient[d] += score_gradient * query_tile[q * head_dimension + d];
                value_gradient[d] += probability * upstream_tile[q * head_dimension + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint key_row = group_start + key_index;
    for (uint d = 0; d < head_dimension; ++d) {
        const float key_total = reduce_sum_across_lanes(key_gradient[d]);
        const float value_total = reduce_sum_across_lanes(value_gradient[d]);
        if (active && lane == 0) {
            key_grad[key_row * head_dimension + d] += key_total;
            value_grad[key_row * head_dimension + d] += value_total;
        }
    }
}

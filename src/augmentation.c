#include "augmentation.h"

#include <math.h>
#include <string.h>

#define AUGMENTATION_PI_F 3.14159265358979323846f

static uint64_t next_random_bits(uint64_t *state) {
    uint64_t mixed = (*state += 0x9E3779B97F4A7C15ULL);
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

static float random_unit(uint64_t *state) {
    return ((float)(next_random_bits(state) >> 40) + 0.5f) / 16777216.0f;
}

static float random_symmetric(uint64_t *state, float magnitude) {
    return magnitude * (2.0f * random_unit(state) - 1.0f);
}

static float random_standard_normal(uint64_t *state) {
    const float radius = sqrtf(-2.0f * logf(random_unit(state)));
    return radius * cosf(2.0f * AUGMENTATION_PI_F * random_unit(state));
}

static size_t random_index_below(uint64_t *state, size_t limit) {
    return (size_t)(next_random_bits(state) % (uint64_t)limit);
}

static float clamp_float(float value, float minimum, float maximum) {
    return value < minimum ? minimum : value > maximum ? maximum : value;
}

static float sample_bilinear_with_edge_padding(const float *plane,
                                               size_t height, size_t width,
                                               float source_y, float source_x) {
    const float clamped_y = clamp_float(source_y, 0.0f, (float)(height - 1));
    const float clamped_x = clamp_float(source_x, 0.0f, (float)(width - 1));
    const size_t top = (size_t)clamped_y;
    const size_t left = (size_t)clamped_x;
    const size_t bottom = top + 1 < height ? top + 1 : top;
    const size_t right = left + 1 < width ? left + 1 : left;
    const float vertical_weight = clamped_y - (float)top;
    const float horizontal_weight = clamped_x - (float)left;
    const float upper = plane[top * width + left] * (1.0f - horizontal_weight) +
                        plane[top * width + right] * horizontal_weight;
    const float lower = plane[bottom * width + left] * (1.0f - horizontal_weight) +
                        plane[bottom * width + right] * horizontal_weight;
    return upper * (1.0f - vertical_weight) + lower * vertical_weight;
}

static int geometric_transform_enabled(const DataAugmentation *augmentation) {
    return augmentation->max_shift_pixels > 0.0f ||
           augmentation->max_rotation_degrees > 0.0f ||
           augmentation->max_scale_delta > 0.0f;
}

static void apply_affine_transform(const float *source, float *destination,
                                   size_t channels, size_t height,
                                   size_t width, float shift_y, float shift_x,
                                   float angle, float scale) {
    const float inverse_cosine = cosf(angle) / scale;
    const float inverse_sine = sinf(angle) / scale;
    const float center_y = 0.5f * (float)(height - 1);
    const float center_x = 0.5f * (float)(width - 1);
    for (size_t channel = 0; channel < channels; ++channel) {
        const float *source_plane = source + channel * height * width;
        float *destination_plane = destination + channel * height * width;
        for (size_t y = 0; y < height; ++y) {
            const float centered_y = (float)y - center_y - shift_y;
            for (size_t x = 0; x < width; ++x) {
                const float centered_x = (float)x - center_x - shift_x;
                const float source_x = inverse_cosine * centered_x +
                                       inverse_sine * centered_y + center_x;
                const float source_y = -inverse_sine * centered_x +
                                       inverse_cosine * centered_y + center_y;
                destination_plane[y * width + x] =
                    sample_bilinear_with_edge_padding(source_plane, height,
                                                      width, source_y,
                                                      source_x);
            }
        }
    }
}

static void apply_geometric_transform(const DataAugmentation *augmentation,
                                      uint64_t *random_state,
                                      const float *source, float *destination,
                                      size_t channels, size_t height,
                                      size_t width) {
    const float shift_y =
        random_symmetric(random_state, augmentation->max_shift_pixels);
    const float shift_x =
        random_symmetric(random_state, augmentation->max_shift_pixels);
    const float angle = random_symmetric(
        random_state, augmentation->max_rotation_degrees) *
        AUGMENTATION_PI_F / 180.0f;
    const float scale =
        1.0f + random_symmetric(random_state, augmentation->max_scale_delta);
    apply_affine_transform(source, destination, channels, height, width,
                           shift_y, shift_x, angle, scale);
}

static void apply_intensity_jitter(const DataAugmentation *augmentation,
                                   uint64_t *random_state, float *image,
                                   size_t channels, size_t plane_size) {
    const float contrast =
        1.0f + random_symmetric(random_state, augmentation->max_contrast_delta);
    const float brightness =
        random_symmetric(random_state, augmentation->max_brightness_delta);
    for (size_t channel = 0; channel < channels; ++channel) {
        float *plane = image + channel * plane_size;
        float plane_mean = 0.0f;
        for (size_t i = 0; i < plane_size; ++i) {
            plane_mean += plane[i];
        }
        plane_mean /= (float)plane_size;
        for (size_t i = 0; i < plane_size; ++i) {
            plane[i] = plane_mean + (plane[i] - plane_mean) * contrast +
                       brightness;
        }
    }
}

static void apply_gaussian_noise(const DataAugmentation *augmentation,
                                 uint64_t *random_state, float *image,
                                 size_t image_size) {
    for (size_t i = 0; i < image_size; ++i) {
        image[i] += augmentation->noise_standard_deviation *
                    random_standard_normal(random_state);
    }
}

static void apply_random_erasing(const DataAugmentation *augmentation,
                                 uint64_t *random_state, float *image,
                                 size_t channels, size_t height, size_t width) {
    if (random_unit(random_state) >= augmentation->erasing_probability) {
        return;
    }
    const size_t max_erased_height = (size_t)fmaxf(
        1.0f, augmentation->erasing_max_side_fraction * (float)height);
    const size_t max_erased_width = (size_t)fmaxf(
        1.0f, augmentation->erasing_max_side_fraction * (float)width);
    const size_t erased_height = 1 + random_index_below(random_state,
                                                        max_erased_height);
    const size_t erased_width = 1 + random_index_below(random_state,
                                                       max_erased_width);
    const size_t top = random_index_below(random_state,
                                          height - erased_height + 1);
    const size_t left = random_index_below(random_state,
                                           width - erased_width + 1);
    for (size_t channel = 0; channel < channels; ++channel) {
        float *plane = image + channel * height * width;
        for (size_t y = top; y < top + erased_height; ++y) {
            for (size_t x = left; x < left + erased_width; ++x) {
                plane[y * width + x] = augmentation->erasing_value;
            }
        }
    }
}

int data_augmentation_enabled(const DataAugmentation *augmentation) {
    return augmentation &&
           (geometric_transform_enabled(augmentation) ||
            augmentation->max_brightness_delta > 0.0f ||
            augmentation->max_contrast_delta > 0.0f ||
            augmentation->noise_standard_deviation > 0.0f ||
            augmentation->erasing_probability > 0.0f);
}

int data_augmentation_apply(const DataAugmentation *augmentation,
                            uint64_t *random_state, const float *source,
                            float *destination, size_t batch, size_t channels,
                            size_t height, size_t width) {
    if (!augmentation || !random_state || !source || !destination ||
        batch == 0 || channels == 0 || height == 0 || width == 0 ||
        augmentation->max_shift_pixels < 0.0f ||
        augmentation->max_rotation_degrees < 0.0f ||
        augmentation->max_scale_delta < 0.0f ||
        augmentation->max_scale_delta >= 1.0f ||
        augmentation->max_brightness_delta < 0.0f ||
        augmentation->max_contrast_delta < 0.0f ||
        augmentation->max_contrast_delta >= 1.0f ||
        augmentation->noise_standard_deviation < 0.0f ||
        augmentation->erasing_probability < 0.0f ||
        augmentation->erasing_probability > 1.0f ||
        augmentation->erasing_max_side_fraction < 0.0f ||
        augmentation->erasing_max_side_fraction > 1.0f ||
        (geometric_transform_enabled(augmentation) && source == destination)) {
        return -1;
    }
    const size_t plane_size = height * width;
    const size_t image_size = channels * plane_size;
    const int clamps_output =
        augmentation->minimum_value < augmentation->maximum_value;
    for (size_t sample = 0; sample < batch; ++sample) {
        const float *source_image = source + sample * image_size;
        float *destination_image = destination + sample * image_size;
        if (geometric_transform_enabled(augmentation)) {
            apply_geometric_transform(augmentation, random_state, source_image,
                                      destination_image, channels, height,
                                      width);
        } else if (destination_image != source_image) {
            memmove(destination_image, source_image,
                    image_size * sizeof(*destination_image));
        }
        if (augmentation->max_brightness_delta > 0.0f ||
            augmentation->max_contrast_delta > 0.0f) {
            apply_intensity_jitter(augmentation, random_state,
                                   destination_image, channels, plane_size);
        }
        if (augmentation->noise_standard_deviation > 0.0f) {
            apply_gaussian_noise(augmentation, random_state, destination_image,
                                 image_size);
        }
        for (size_t i = 0; clamps_output && i < image_size; ++i) {
            destination_image[i] = clamp_float(destination_image[i],
                                               augmentation->minimum_value,
                                               augmentation->maximum_value);
        }
        if (augmentation->erasing_probability > 0.0f) {
            apply_random_erasing(augmentation, random_state, destination_image,
                                 channels, height, width);
        }
    }
    return 0;
}

int data_augmentation_translate(const float *source, float *destination,
                                size_t batch, size_t channels, size_t height,
                                size_t width, float shift_y, float shift_x) {
    if (!source || !destination || source == destination || batch == 0 ||
        channels == 0 || height == 0 || width == 0) {
        return -1;
    }
    const size_t image_size = channels * height * width;
    for (size_t sample = 0; sample < batch; ++sample) {
        apply_affine_transform(source + sample * image_size,
                               destination + sample * image_size, channels,
                               height, width, shift_y, shift_x, 0.0f, 1.0f);
    }
    return 0;
}

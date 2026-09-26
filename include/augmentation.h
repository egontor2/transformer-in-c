#ifndef AUGMENTATION_H
#define AUGMENTATION_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float max_shift_pixels;
    float max_rotation_degrees;
    float max_scale_delta;
    float max_brightness_delta;
    float max_contrast_delta;
    float noise_standard_deviation;
    float erasing_probability;
    float erasing_max_side_fraction;
    float erasing_value;
    float minimum_value;
    float maximum_value;
} DataAugmentation;

int data_augmentation_enabled(const DataAugmentation *augmentation);

int data_augmentation_apply(const DataAugmentation *augmentation,
                            uint64_t *random_state, const float *source,
                            float *destination, size_t batch, size_t channels,
                            size_t height, size_t width);

int data_augmentation_translate(const float *source, float *destination,
                                size_t batch, size_t channels, size_t height,
                                size_t width, float shift_y, float shift_x);

#endif

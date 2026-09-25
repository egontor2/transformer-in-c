#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

typedef struct {
    float *images;
    size_t *labels;
    size_t sample_count;
    size_t height;
    size_t width;
    size_t channels;
} VisionDataset;

/*
 * Loads a manifest whose lines have the form:
 *   path/to/image.pgm,label
 *
 * PGM P2 (ASCII) and P5 (binary) images are supported. Samples must share
 * dimensions; pixels are returned as floats normalized to [0, 1].
 */
int vision_dataset_load_pgm_csv(const char *manifest_path,
                                VisionDataset *dataset);
void vision_dataset_free(VisionDataset *dataset);

#endif

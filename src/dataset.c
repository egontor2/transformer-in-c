#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_pgm_token(FILE *file, char *buffer, size_t capacity) {
    int character;
    do {
        character = fgetc(file);
    } while (character != EOF && (character == ' ' || character == '\n' ||
                                  character == '\r' || character == '\t'));
    if (character == '#') {
        do {
            character = fgetc(file);
        } while (character != EOF && character != '\n');
        return read_pgm_token(file, buffer, capacity);
    }
    if (character == EOF) {
        return -1;
    }
    size_t length = 0;
    while (character != EOF && character != ' ' && character != '\n' &&
           character != '\r' && character != '\t') {
        if (length + 1 >= capacity) {
            return -1;
        }
        buffer[length++] = (char)character;
        character = fgetc(file);
    }
    buffer[length] = '\0';
    return 0;
}

static int parse_size(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || parsed == 0) {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int parse_nonnegative(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (text[0] == '\0' || *end != '\0') {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int load_pgm(const char *path, float **pixels, size_t *height,
                    size_t *width) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return -1;
    }
    char token[64];
    size_t max_value;
    if (read_pgm_token(file, token, sizeof(token)) != 0 ||
        (strcmp(token, "P2") != 0 && strcmp(token, "P5") != 0)) {
        fclose(file);
        return -1;
    }
    const int binary = strcmp(token, "P5") == 0;
    if (read_pgm_token(file, token, sizeof(token)) != 0 ||
        parse_size(token, width) != 0 ||
        read_pgm_token(file, token, sizeof(token)) != 0 ||
        parse_size(token, height) != 0 ||
        read_pgm_token(file, token, sizeof(token)) != 0 ||
        parse_size(token, &max_value) != 0 || max_value > 65535) {
        fclose(file);
        return -1;
    }
    const size_t count = *width * *height;
    float *data = malloc(count * sizeof(float));
    if (!data) {
        fclose(file);
        return -1;
    }
    if (binary) {
        for (size_t i = 0; i < count; ++i) {
            unsigned int value;
            if (max_value < 256) {
                value = (unsigned int)fgetc(file);
                if (value == (unsigned int)EOF) {
                    free(data);
                    fclose(file);
                    return -1;
                }
            } else {
                const int high = fgetc(file);
                const int low = fgetc(file);
                if (high == EOF || low == EOF) {
                    free(data);
                    fclose(file);
                    return -1;
                }
                value = ((unsigned int)high << 8) | (unsigned int)low;
            }
            data[i] = (float)value / (float)max_value;
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            if (read_pgm_token(file, token, sizeof(token)) != 0) {
                free(data);
                fclose(file);
                return -1;
            }
            size_t value;
            if (parse_nonnegative(token, &value) != 0 || value > max_value) {
                free(data);
                fclose(file);
                return -1;
            }
            data[i] = (float)value / (float)max_value;
        }
    }
    fclose(file);
    *pixels = data;
    return 0;
}

void vision_dataset_free(VisionDataset *dataset) {
    if (!dataset) {
        return;
    }
    free(dataset->images);
    free(dataset->labels);
    memset(dataset, 0, sizeof(*dataset));
}

int vision_dataset_load_pgm_csv(const char *manifest_path,
                                VisionDataset *dataset) {
    if (!manifest_path || !dataset) {
        return -1;
    }
    memset(dataset, 0, sizeof(*dataset));
    FILE *manifest = fopen(manifest_path, "r");
    if (!manifest) {
        return -1;
    }
    size_t capacity = 8;
    dataset->images = malloc(capacity * sizeof(float));
    dataset->labels = malloc(capacity * sizeof(size_t));
    if (!dataset->images || !dataset->labels) {
        fclose(manifest);
        vision_dataset_free(dataset);
        return -1;
    }
    char line[4096];
    while (fgets(line, sizeof(line), manifest)) {
        char path[2048];
        unsigned long label;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sscanf(line, " %2047[^,],%lu", path, &label) != 2) {
            fclose(manifest);
            vision_dataset_free(dataset);
            return -1;
        }
        float *image = NULL;
        size_t height = 0;
        size_t width = 0;
        if (load_pgm(path, &image, &height, &width) != 0) {
            fclose(manifest);
            vision_dataset_free(dataset);
            return -1;
        }
        if (dataset->sample_count == 0) {
            dataset->height = height;
            dataset->width = width;
        } else if (dataset->height != height || dataset->width != width) {
            free(image);
            fclose(manifest);
            vision_dataset_free(dataset);
            return -1;
        }
        const size_t image_size = height * width;
        if (dataset->sample_count == 0) {
            free(dataset->images);
            dataset->images = malloc(capacity * image_size * sizeof(float));
        } else if (dataset->sample_count == capacity) {
            capacity *= 2;
            float *resized_images = realloc(dataset->images,
                                            capacity * image_size * sizeof(float));
            size_t *resized_labels = realloc(dataset->labels,
                                             capacity * sizeof(size_t));
            if (!resized_images || !resized_labels) {
                if (resized_images) {
                    dataset->images = resized_images;
                }
                if (resized_labels) {
                    dataset->labels = resized_labels;
                }
                free(image);
                fclose(manifest);
                vision_dataset_free(dataset);
                return -1;
            }
            dataset->images = resized_images;
            dataset->labels = resized_labels;
        }
        if (!dataset->images) {
            free(image);
            fclose(manifest);
            vision_dataset_free(dataset);
            return -1;
        }
        memcpy(dataset->images + dataset->sample_count * image_size, image,
               image_size * sizeof(float));
        dataset->labels[dataset->sample_count++] = (size_t)label;
        free(image);
    }
    fclose(manifest);
    if (dataset->sample_count == 0) {
        vision_dataset_free(dataset);
        return -1;
    }
    dataset->channels = 1;
    return 0;
}

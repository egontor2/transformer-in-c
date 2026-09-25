#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

typedef struct {
    unsigned char *memory;
    size_t capacity;
    size_t offset;
} Arena;

int arena_init(Arena *arena, size_t capacity);
void arena_reset(Arena *arena);
void arena_free(Arena *arena);
void *arena_alloc(Arena *arena, size_t size, size_t alignment);

#endif

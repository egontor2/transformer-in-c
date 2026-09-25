#include "arena.h"

#include <stdint.h>
#include <stdlib.h>

int arena_init(Arena *arena, size_t capacity) {
    if (!arena || capacity == 0) {
        return -1;
    }
    arena->memory = malloc(capacity);
    arena->capacity = capacity;
    arena->offset = 0;
    if (!arena->memory) {
        arena->capacity = 0;
        return -1;
    }
    return 0;
}

void arena_reset(Arena *arena) {
    if (arena) {
        arena->offset = 0;
    }
}

void arena_free(Arena *arena) {
    if (!arena) {
        return;
    }
    free(arena->memory);
    arena->memory = NULL;
    arena->capacity = 0;
    arena->offset = 0;
}

void *arena_alloc(Arena *arena, size_t size, size_t alignment) {
    if (!arena || !arena->memory || size == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return NULL;
    }
    const uintptr_t address = (uintptr_t)arena->memory + arena->offset;
    const size_t padding = (size_t)((alignment - (address & (alignment - 1))) &
                                    (alignment - 1));
    if (padding > arena->capacity - arena->offset ||
        size > arena->capacity - arena->offset - padding) {
        return NULL;
    }
    arena->offset += padding;
    void *result = arena->memory + arena->offset;
    arena->offset += size;
    return result;
}

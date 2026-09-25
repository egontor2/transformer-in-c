CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
CPPFLAGS := -Iinclude
LDFLAGS := -lm

LIBRARY := build/libtransformer.a
OBJECTS := build/transformer.o build/dataset.o build/autodiff.o build/arena.o build/ops.o

.PHONY: all clean run test

all: $(LIBRARY) build/basic

$(LIBRARY): $(OBJECTS)
	@mkdir -p $(@D)
	ar rcs $@ $^

build/transformer.o: src/transformer.c include/transformer.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/dataset.o: src/dataset.c include/dataset.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/autodiff.o: src/autodiff.c include/autodiff.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/arena.o: src/arena.c include/arena.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/ops.o: src/ops.c include/ops.h include/tensor.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/basic: examples/basic.c $(LIBRARY)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) $(LDFLAGS) -o $@

run: build/basic
	./build/basic

test: build/test_vision
	./build/test_vision

build/test_vision: tests/test_vision.c $(LIBRARY)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) $(LDFLAGS) -o $@

clean:
	rm -rf build

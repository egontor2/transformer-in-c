CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
CPPFLAGS := -Iinclude
LDFLAGS := -lm

LIBRARY := build/libtransformer.a
OBJECTS := build/transformer.o build/dataset.o build/autodiff.o build/arena.o build/ops.o build/vit.o

.PHONY: all clean run test organ-smnist organ-smnist-mps mps-test

all: $(LIBRARY) build/basic build/organ_smnist

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

build/vit.o: src/vit.c include/vit.h include/tensor.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/basic: examples/basic.c $(LIBRARY)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) $(LDFLAGS) -o $@

build/organ_smnist_mps: examples/organ_smnist.c $(LIBRARY) build/mps_backend.o
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DENABLE_MPS $< $(LIBRARY) \
		build/mps_backend.o -framework Foundation -framework Metal \
		-framework MetalPerformanceShaders -lm -lobjc -lstdc++ -o $@

build/organ_smnist: examples/organ_smnist.c $(LIBRARY)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) $(LDFLAGS) -o $@

run: build/basic
	./build/basic

organ-smnist: build/organ_smnist
	./build/organ_smnist

organ-smnist-mps: build/organ_smnist_mps
	./build/organ_smnist_mps

mps-test: build/mps_test
	./build/mps_test

build/mps_backend.o: src/mps_backend.mm include/mps_backend.h
	@mkdir -p $(@D)
	clang++ -Iinclude -std=c++17 -fobjc-arc -c $< -o $@

build/mps_test: examples/mps_test.c $(LIBRARY) build/mps_backend.o
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) build/mps_backend.o \
		-framework Foundation -framework Metal \
		-framework MetalPerformanceShaders -lm -lobjc -lstdc++ -o $@

test: build/test_vision
	./build/test_vision

build/test_vision: tests/test_vision.c $(LIBRARY)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) $(LDFLAGS) -o $@

clean:
	rm -rf build

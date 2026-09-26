CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
CPPFLAGS := -Iinclude
LDFLAGS := -lm

NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= native
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(CUDA_ARCH)
CUDA_LDFLAGS := -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ -lm

LIBRARY := build/libtransformer.a
OBJECTS := build/transformer.o build/dataset.o build/autodiff.o build/arena.o build/ops.o build/vit.o build/augmentation.o

.PHONY: all clean run test organ-smnist organ-smnist-mps organ-smnist-cuda mps-test metal-test cuda-test

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

build/augmentation.o: src/augmentation.c include/augmentation.h
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

build/metal_kernels.inc: src/metal_kernels.metal
	@mkdir -p $(@D)
	cd src && xxd -i metal_kernels.metal > ../$@

build/mps_backend.o: src/mps_backend.mm include/mps_backend.h include/ops.h \
		include/tensor.h build/metal_kernels.inc
	@mkdir -p $(@D)
	clang++ -Iinclude -Ibuild -std=c++17 -fobjc-arc -O2 -Wall -Wextra -c $< -o $@

build/mps_test: examples/mps_test.c $(LIBRARY) build/mps_backend.o
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY) build/mps_backend.o \
		-framework Foundation -framework Metal \
		-framework MetalPerformanceShaders -lm -lobjc -lstdc++ -o $@

metal-test: build/test_metal
	./build/test_metal

build/test_metal: tests/test_device.c $(LIBRARY) build/mps_backend.o
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

build/cuda_backend.o: src/cuda_backend.cu include/cuda_backend.h include/ops.h \
		include/tensor.h
	@mkdir -p $(@D)
	$(NVCC) -Iinclude $(NVCCFLAGS) -c $< -o $@

build/organ_smnist_cuda: examples/organ_smnist.c $(LIBRARY) build/cuda_backend.o
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DENABLE_CUDA $< $(LIBRARY) \
		build/cuda_backend.o $(CUDA_LDFLAGS) -o $@

build/test_cuda: tests/test_device.c $(LIBRARY) build/cuda_backend.o
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DENABLE_CUDA $< $(LIBRARY) \
		build/cuda_backend.o $(CUDA_LDFLAGS) -o $@

organ-smnist-cuda: build/organ_smnist_cuda
	./build/organ_smnist_cuda

cuda-test: build/test_cuda
	./build/test_cuda

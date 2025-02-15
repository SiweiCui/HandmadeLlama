# include "alloc.hpp"
# include <stdio.h>

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, byte_size);
    if(err != cudaSuccess) {
      printf("CUDADeviceAllocator::allocate(): cudaMalloc failed\n");
      return nullptr;
    }
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    cudaError_t err = cudaFree(ptr);
    if(err != cudaSuccess) {
      printf("CUDADeviceAllocator::release(): cudaFree failed\n");
    }
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
}
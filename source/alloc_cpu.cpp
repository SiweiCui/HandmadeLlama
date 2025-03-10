//
// Created by CSWH on 2024/10/28.
//
#include "alloc.hpp"
#include <cstdlib>

namespace base {
// CPU
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if(!byte_size) {
      return nullptr;
    }
    void* data = malloc(byte_size);
    return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
    if(ptr){
      free(ptr);
    }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}// namespace base


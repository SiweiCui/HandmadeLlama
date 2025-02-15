//
// Created by CSWH on 2024/11/17.
//
#include "alloc.hpp"

namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (!byte_size) {
        return;
    }

    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size) const {
    CHECK(device_type_ != DeviceType::kDeviceUnknown);
    if (device_type_ == DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        cudaMemset(ptr, 0, byte_size);
    }
}
}
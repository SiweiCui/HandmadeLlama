//
// Created by CSWH on 2024/11/17.
//
# include "buffer.hpp"
# include <glog/logging.h>
#include <stdio.h>

namespace base{
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
           void* ptr, bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
    if(!ptr_ && allocator_){ // 如果没有ptr, 有allocator, 说明要自己分配且自己管理.
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
    }
}

Buffer::~Buffer() {
    if(!use_external_){
        if(ptr_ && allocator_){
          allocator_->release(ptr_);
          ptr_ = nullptr;
        }
    }
    // LOG(INFO) << "Buffer::~Buffer()";
    // printf("内存已经释放\n");
}

size_t Buffer::byte_size() const {
    return byte_size_;
}
const void* Buffer::ptr() const {
    return ptr_;
}

void* Buffer::ptr() {
    return ptr_;
}

DeviceType Buffer::device_type() const {
    return device_type_;
}

void Buffer::set_device_type(DeviceType device_type) {
    device_type_ = device_type;
}

bool Buffer::is_external() const {
    return this->use_external_;
}

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer->ptr() != nullptr);

    size_t byte_size = byte_size_ < buffer->byte_size() ? byte_size_ : buffer->byte_size();

    const DeviceType buffer_device = buffer->device_type();
    const DeviceType current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown && current_device != DeviceType::kDeviceUnknown);

    if(buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU){
        return allocator_->memcpy(buffer->ptr(), ptr_, byte_size, MemcpyKind::kMemcpyCPU2CPU);
    }
    if(buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCPU){
        return allocator_->memcpy(buffer->ptr(), ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
    }
    if(buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCUDA){
        return allocator_->memcpy(buffer->ptr(), ptr_, byte_size, MemcpyKind::kMemcpyCPU2CUDA);
    }
    if(buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCUDA){
        return allocator_->memcpy(buffer->ptr(), ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CUDA);
    }
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
    return allocator_;
}


}
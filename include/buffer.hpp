//
// Created by CSWH on 2024/11/17.
//

#ifndef BUFFER_HPP
#define BUFFER_HPP
#include <memory>
#include "alloc.hpp"


namespace base {
  /*
  1. 自己分配自己管理
  2. 不是自己分配, 但需要管理(需要管理就需要allocator)
  3. 不是自己分配, 也不需要管理
 */
  class Buffer{
    private:
      size_t byte_size_ = 0;
      void* ptr_ = nullptr;
      bool use_external_ = false;
      DeviceType device_type_ = DeviceType::kDeviceUnknown;
      std::shared_ptr<DeviceAllocator> allocator_; // 父类类型指向子类实现

    public:
      // 只传byte_size和allocator, 会在构造函数中自动分配内存并管理.
      explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
             void* ptr = nullptr, bool use_external = false);

      virtual ~Buffer(); // 虚析构

      const void* ptr() const;

      void* ptr();

      size_t byte_size() const;

      bool is_external() const;

      DeviceType device_type() const;

      void set_device_type(DeviceType device_type);

      void copy_from(const Buffer* buffer) const;

      std::shared_ptr<DeviceAllocator> allocator() const;

  };
} // namespace base
#endif //BUFFER_HPP

//
// Created by CSWH on 2024/11/17.
//
#include <gtest/gtest.h>
#include "buffer.hpp"
#include <memory>

TEST(test_buffer, all_kinds_of_buffer) {
  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto gpu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  {
    // 1. 自己分配, 自己管理
    auto buffer = std::make_shared<base::Buffer>(1024,
                                                 gpu_alloc,
                                                 nullptr,
                                                 true
                                                 );
    ASSERT_NE(buffer->ptr(), nullptr);
  }

}
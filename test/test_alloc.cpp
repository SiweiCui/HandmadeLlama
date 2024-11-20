//
// Created by CSWH on 2024/11/16.
//
#include <gtest/gtest.h>
#include "alloc.hpp"

TEST(test_alloc, allocate) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    void* ptr = alloc->allocate(1024);
    ASSERT_NE(ptr, nullptr);
    alloc->release(ptr);
}

TEST(test_alloc, allocate_gpu) {
    auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
    void* ptr = alloc->allocate(1024);
    ASSERT_NE(ptr, nullptr);
    alloc->release(ptr);
}
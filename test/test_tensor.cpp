//
// Created by CSWH on 2024/11/17.
//

#include <gtest/gtest.h>
#include "tensor.hpp"
#include <memory>

TEST(test_tensor, init_tensor) {
    auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor tensor(base::DataType::kDataTypeInt8,
        32, 2048, 2048,
        true,
        alloc_gpu,
        nullptr);
	ASSERT_EQ(tensor.is_empty(), false);
}

TEST(test_tensor, data_from_out) {
	auto* data = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * 32));
	for (int i = 0; i < 32; i++) {
		data[i] = i;
	}
	ASSERT_EQ(data[5], 5);
	auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
	auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();

	// 外部数据是管不了的.
	tensor::Tensor tensor(base::DataType::kDataTypeInt32,
		2, 2, 8,
		false, nullptr, data);
	ASSERT_EQ(tensor.is_empty(), false);
	auto& mydata = tensor.index<int32_t>(5);
	ASSERT_EQ(mydata, 5);

	tensor.set_device_type(base::DeviceType::kDeviceCPU);
	tensor.to_cuda();// 这时候能管了.
	ASSERT_EQ(tensor.device_type(), base::DeviceType::kDeviceCUDA);
}
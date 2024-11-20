//
// Created by CSWH on 2024/11/19.
//
# include "add.cuh"
#include "../../../../usr/local/cuda-11.6/targets/x86_64-linux/include/cuda_runtime.h"
#include <memory>

/*
 *	两种使用方法
 */
TEST(test_add, test_add_layer) {
	// Tensor就一种使用形式: 创建空, 再assign
	tensor::Tensor inTensor1(base::DataType::kDataTypeFp32, 128);
	tensor::Tensor inTensor2(base::DataType::kDataTypeFp32, 128);
	tensor::Tensor outTensor(base::DataType::kDataTypeFp32, 128);

	// 创建数据
	auto in1 = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	auto in2 = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	auto out = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	for (int i = 0; i < 128; i++) {
		in1[i] = static_cast<float>(i);
		in2[i] = static_cast<float>(i);
	}

	// 转移到gpu上
	float* in1_device, *in2_device, *out_device;
	cudaMalloc((void**)&in1_device, sizeof(float) * 128);
	cudaMalloc((void**)&in2_device, sizeof(float) * 128);
	cudaMalloc((void**)&out_device, sizeof(float) * 128);
	cudaMemcpy(in1_device, in1, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(in2_device, in2, sizeof(float) * 128, cudaMemcpyHostToDevice);

	// 创建buffer并assign数据
	auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
	auto in1_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cu, in1_device); // 这样buffer会管理的.
	auto in2_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cu, in2_device);
	auto out_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cu, out_device);
	inTensor1.assign(in1_buffer);
	inTensor2.assign(in2_buffer);
	outTensor.assign(out_buffer);

	// 创建算子层进行计算
	auto add_layer = op::VecAddLayer();
	add_layer.set_input(0, inTensor1);
	add_layer.set_input(1, inTensor2);
	add_layer.set_output(0, outTensor);
	add_layer.forward(); // forward默认调用核函数, 不用考虑设备问题

	// 移动回数据到Host
	cudaMemcpy(out, out_device, sizeof(float) * 128, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 128; i++) {
		printf("%f\n", out[i]);
	}

	free(in1);
	free(in2);
	free(out);
}


TEST(test_add, test_add_layer_tidy) {
	// Tensor就一种使用形式: 创建空, 再assign
	tensor::Tensor inTensor1(base::DataType::kDataTypeFp32, 128);
	tensor::Tensor inTensor2(base::DataType::kDataTypeFp32, 128);
	tensor::Tensor outTensor(base::DataType::kDataTypeFp32, 128);

	// 创建数据
	auto in1 = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	auto in2 = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	auto out = reinterpret_cast<float*>(malloc(128 * sizeof(float)));
	for (int i = 0; i < 128; i++) {
		in1[i] = static_cast<float>(i);
		in2[i] = static_cast<float>(i);
	}

	// 创建buffer管理, 并assign数据
	auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
	auto in1_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cpu, in1); // 这样buffer会管理的.
	auto in2_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cpu, in2);
	auto out_buffer = std::make_shared<base::Buffer>(sizeof(float) * 128, alloc_cpu, out);

	inTensor1.assign(in1_buffer);
	inTensor2.assign(in2_buffer);
	outTensor.assign(out_buffer);
	inTensor1.set_device_type(base::DeviceType::kDeviceCPU);
	inTensor2.set_device_type(base::DeviceType::kDeviceCPU);
	outTensor.set_device_type(base::DeviceType::kDeviceCPU);

	// 创建算子层进行计算
	auto add_layer = op::VecAddLayer();
	add_layer.set_input(0, inTensor1);
	add_layer.set_input(1, inTensor2);
	add_layer.set_output(0, outTensor);
	add_layer.to_cuda(); // 转移到gpu上
	add_layer.forward(); // forward默认调用核函数, 不用考虑设备问题

	// 移动回数据到Host
	cudaMemcpy(out, add_layer.get_output(0).ptr<float>(), sizeof(float) * 128, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 128; i++) {
		printf("%f\n", out[i]);
	}

}
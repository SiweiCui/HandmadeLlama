//
// Created by CSWH on 2024/11/18.
//

# include "swiglu.cuh"

namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
	: Layer(base::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
	reset_input_size(2);
	reset_output_size(1);
}

bool SwiGLULayer::forward() {
	auto input1 = this->get_input(0);
	auto input2 = this->get_input(1);
	auto output = this->get_output(0);
	kernel::swiglu_kernel_cu(input1, input2, output);
	return true;
}
}

namespace kernel {
// 多block以实现覆盖所有元素, 不涉及规约, 与加法类似
// in1 * sigma(in1) * in2
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= size) {
		return;
	}
	// 外部定义
	extern __shared__ float shared_mem[];
	// 指针运算
	float* smem1 = shared_mem;
	float* smem2 = shared_mem + blockDim.x;

	smem1[tid] = in1[idx];
	smem2[tid] = in2[idx];
	__syncthreads();

	float value = 1.0f / (1.0f + exp(-smem1[tid]));
	smem1[tid] = smem1[tid] * value;
	// 一个线程计算结果的一个元素
	out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
					  const tensor::Tensor& output) {
	CHECK_EQ(input1.is_empty(), false);
	CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

	CHECK_EQ(input2.is_empty(), false);
	CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

	CHECK_EQ(output.is_empty(), false);
	CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

	int size = static_cast<int32_t>(input1.size());
	int threads = 128;
	int blocks = (size + threads - 1) / threads;
	const size_t shmem = threads * sizeof(float) * 2;

	swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
			size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
}
}
//
// Created by CSWH on 2024/11/18.
//
# include "add.cuh"

namespace op {
VecAddLayer::VecAddLayer()
: Layer(base::LayerType::kLayerAdd, "Add") {
	reset_input_size(2);
	reset_output_size(1);
}

// 必须确保向量已经在gpu上
bool VecAddLayer::forward() {
	auto input1 = this->get_input(0);
	auto input2 = this->get_input(1);
	auto output = this->get_output(0);

	kernel::add_kernel_cu(input1, input2, output);
	return true;
}
}

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size) {
		return;
	}
	float in_val1 = in1[tid];
	float in_val2 = in2[tid];
	out[tid] = in_val1 + in_val2;

}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
			   const tensor::Tensor& output, void* stream) {
	CHECK_EQ(input1.is_empty(), false);
	CHECK_EQ(input2.is_empty(), false);
	CHECK_EQ(output.is_empty(), false);
	auto size = static_cast<int32_t>(input1.size());
	CHECK_EQ(size, input2.size());
	CHECK_EQ(size, output.size());
	int32_t thread_num = 512;
	int32_t block_num = (size + thread_num - 1) / thread_num;

	add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
													  const_cast<float*>(output.ptr<float>()));
}
}
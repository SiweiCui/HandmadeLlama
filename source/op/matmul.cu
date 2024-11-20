//
// Created by CSWH on 2024/11/19.
//
# include "matmul.cuh"
#include "add.cuh"
#include <cub/block/block_reduce.cuh>

namespace op {
	MatmulLayer::MatmulLayer(int32_t dim0, int32_t dim1, bool has_bias)
		: LayerParam(base::LayerType::kLayerMatmul, "Matmul"),
		dim0_(dim0),
		dim1_(dim1),
		has_bias_(has_bias){
		reset_input_size(1);
		reset_output_size(1);
		reset_weight_size(1);
		if (has_bias_) {
			bias_.resize(1);
		}
	}

	bool MatmulLayer::forward() {
		kernel::matmul_kernel_cu_qint8(get_input(0), get_weight(0), get_output(0),
											   group_size_, scales_);
		if (has_bias_) {
			kernel::add_kernel_cu(get_output(0), get_bias(0), get_output(0));
		}

		return true;
	}

	// bias设置后会在cpu上
	bool MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr) {
		CHECK_GE(idx, 0);
		CHECK_LT(idx, bias_.size());
		CHECK_NE(bias_ptr, nullptr);

		size_t size = dim * sizeof(float);
		std::shared_ptr<base::Buffer> buffer =
			std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);

		// is quant layer
		tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
		bias.set_device_type(base::DeviceType::kDeviceCPU);
		CHECK(bias.assign(buffer));
		bias_.at(idx) = bias;

		const int32_t bias_size = static_cast<int32_t>(bias.size());
		CHECK(bias_size % group_size_ == 0);

		int32_t scale_nums = bias_size / group_size_;
		scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
									 reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
		scales_.set_device_type(base::DeviceType::kDeviceCPU);

		return true;
	}

	tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
		CHECK_GE(idx, 0);
		CHECK_LT(idx, bias_.size());
		return bias_.at(idx);
	}

	const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
		CHECK_GE(idx, 0);
		CHECK_LT(idx, bias_.size());
		return bias_.at(idx);
	}

	void MatmulLayer::to_cuda() {
		LayerParam::to_cuda();
		if (has_bias_) {
			for (auto& bias : bias_) {
				bias.to_cuda();
			}
		}
	}

}

namespace kernel {
	// SGEMV: 矩阵-向量乘法
	template<int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
	__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
											const float* scale, const int32_t group_size,
											float* output, int M, int K) {
		__shared__ float sdata[THREAD_PER_BLOCK];
		int tidInBlock = threadIdx.x;
		int startRow = blockIdx.x * ROW_PER_BLOCK;
		int endRow = startRow + ROW_PER_BLOCK - 1;
		if (startRow >= K) {return;}
		// K: row, M: col
		for (int row = startRow; row <= endRow; ++row) {
			if (row >= K) {break;}
			sdata[tidInBlock] = 0;
			// input与W的每一行做内积
			int globalRow = row * M;
			for (int i = tidInBlock; i < M; i += THREAD_PER_BLOCK) {
				sdata[tidInBlock] += input[i] * scale[(globalRow + i) / group_size] * static_cast<float>(weight[globalRow+i]);
			}
			__syncthreads();

			// 每一行做一次规约
			using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
			__shared__ typename BlockReduce::TempStorage temp;
			float part_sum = BlockReduce(temp).Sum(sdata[tidInBlock]);
			__syncthreads();

			if (tidInBlock == 0) {
				output[row] = part_sum;
			}

			__syncthreads();
		}

	}

	void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
							const tensor::Tensor& output, int32_t group_size,
							const tensor::Tensor& scale) {
		CHECK(input.is_empty() == false && input.dims_size() <= 2);
		CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

		CHECK(weight.is_empty() == false && weight.dims_size() == 2);
		CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

		const int32_t K = weight.get_dim(0);  // row
		const int32_t M = weight.get_dim(1);  // col

		CHECK_EQ(M, input.get_dim(0));
		// 实际执行时每个block计算一行.
		matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
													  scale.ptr<float>(), group_size,
													  const_cast<float*>(output.ptr<float>()), M, K);
	}


}

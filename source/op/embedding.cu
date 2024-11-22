//
// Created by CSWH on 2024/11/19.
//
# include "embedding.cuh"

namespace op {
	EmbeddingLayer::EmbeddingLayer(int32_t dim, int32_t seq_len,
								   int32_t vocab_size)
		: dim_(dim),
		  seq_len_(seq_len),
		  vocab_size_(vocab_size),
		  LayerParam(base::LayerType::kLayerEmbedding, "Embedding") {
		reset_weight_size(1);
		reset_input_size(2);
		reset_output_size(1);
	}

	bool EmbeddingLayer::forward() {

		kernel::emb_kernel_cu(get_input(0), get_weight(0), get_output(0), vocab_size_);
		return true;
	}
}  // namespace op

namespace kernel {
	__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
									   const int32_t* input_ptr, const float* weight_ptr,
									   float* output_ptr) {
		int32_t token_idx = blockIdx.x;
		if (token_idx >= token_num) {
			return;
		}
		int32_t token = input_ptr[token_idx];
		if (token >= vocab_size) {
			return;
		}

		float* output_ptr_start = output_ptr + token_idx * weight_dim;
		const float* weight_ptr_start = weight_ptr + token * weight_dim;

		for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
			output_ptr_start[i] = weight_ptr_start[i];
		}
	}

	void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
					   const tensor::Tensor& output, int32_t vocab_size) {
		tensor::Tensor input_cu;
		if (input.device_type() != base::DeviceType::kDeviceCUDA) {
			input_cu = input.clone();
			input_cu.to_cuda();
		}
		const int32_t input_num = static_cast<int32_t>(input.size());
		const int32_t weight_dim = weight.get_dim(1);
		CHECK(weight.device_type() == output.device_type());
		CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

		constexpr int32_t max_seq_len = 512;
		constexpr int32_t thread_num = 128;
		int32_t* in_ptr = input_cu.ptr<int32_t>();
		float* wei_ptr = const_cast<float*>(weight.ptr<float>());
		float* out_ptr = const_cast<float*>(output.ptr<float>());
		emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
															wei_ptr, out_ptr);
	}
}
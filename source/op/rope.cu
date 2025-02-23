//
// Created by CSWH on 2024/11/19.
//
# include "rope.cuh"

namespace op {
RoPELayer::RoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size)
	: Layer(base::LayerType::kLayerRoPe, "RoPe"),
	  dim_(dim),
	  kv_dim_(kv_dim),
	  head_size_(head_size) {
	reset_input_size(5);
	reset_output_size(1);
}

bool RoPELayer::forward() {
	tensor::Tensor input_q = this->get_input(0);
	tensor::Tensor input_k = this->get_input(1);
	tensor::Tensor input_pos = this->get_input(2);

	tensor::Tensor sin_cache = this->get_input(3);
	tensor::Tensor cos_cache = this->get_input(4);

	kernel::rope_kernel_cu(dim_, kv_dim_, head_size_, input_q, input_k, input_pos,
										  sin_cache, cos_cache);
	return true;
}

}  // namespace op

namespace kernel {
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
	float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
	float2 vec_value = *vec_ptr;
	*vec_ptr =
		make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
									const float* input_q, const float* input_k,
									const float* sin_cache, const float* cos_cache) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x; // 全局线程索引
	// 0号线程负责0和1, 1负责2和3, ...,
	idx = idx * 2;
	if (idx >= dim) { // 超出则跳过.
		return;
	}

	int head_dim = idx % head_size;
	float fci = *(sin_cache + pos * head_size + head_dim);
	float fcr = *(cos_cache + pos * head_size + head_dim);

	rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
	if (idx >= kv_dim) {
		return;
	}
	rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                  const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                  const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) {
	const int32_t pos = *input_pos.ptr<int32_t>(0);
	int threads = 128;
	int blocks = (dim + threads - 1) / threads;
	rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
	                                           input_k.ptr<float>(), sin_cache.ptr<float>(),
	                                           cos_cache.ptr<float>());

}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int head_dim = idx % head_size;
	for (int pos = 0; pos < max_seq_len; ++pos) {
		float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
		float val = static_cast<float>(pos) * freq;
		float fcr = cosf(val);
		float fci = sinf(val);
		*(sin_cache + pos * head_size + head_dim) = fci;
		*(cos_cache + pos * head_size + head_dim) = fcr;
	}
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
						 const tensor::Tensor& cos_cache) {
	CHECK_EQ(sin_cache.is_empty(), false);
	CHECK_EQ(cos_cache.is_empty(), false);
	int threads = head_size;
	sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
									 const_cast<float*>(cos_cache.ptr<float>()));
}
}
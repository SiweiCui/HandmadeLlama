//
// Created by CSWH on 2024/11/19.
//

#ifndef MHA_CUH
#define MHA_CUH
#include "layer.hpp"

namespace op {
	class MultiHeadAttention : public Layer {
	public:
		explicit MultiHeadAttention(int32_t layer_index,
									int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
									int32_t head_num, int32_t head_size);

		void set_pos(int32_t pos);
		void set_layer_idx(int32_t layer_idx);

		bool forward() override;

	private:
		int32_t layer_index_ = 0;
		int32_t pos_ = 0;
		int32_t kv_mul_ = 0;
		int32_t kv_dim_ = 0;
		int32_t seq_len_ = 0;
		int32_t head_num_ = 0;
		int32_t head_size_ = 0;
	};
}  // namespace op

namespace kernel {
	void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
					   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
					   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
					   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor);
	__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
											float* score_ptr, float* output, float* key_cache,
											float* value_cache, int32_t kv_dim, int32_t kv_mul,
											int32_t head_num, int32_t head_size,
											int32_t layer_offset);

	__global__ void flash_attention_kernel(int32_t pos, int32_t seq_len, float* query,
												float* score_ptr, float* output, float* key_cache,
												float* value_cache, int32_t kv_dim, int32_t kv_mul,
												int32_t head_num, int32_t head_size,
												int32_t layer_offset);
}
#endif //MHA_CUH

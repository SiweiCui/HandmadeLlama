//
// Created by CSWH on 2024/11/19.
//

#ifndef ROPE_CUH
#define ROPE_CUH
#include "layer.hpp"

namespace op {
class RoPELayer : public Layer {
public:
	explicit RoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size);

	bool forward() override;
	using Layer::forward;

private:
	int32_t dim_ = 0;
	int32_t kv_dim_ = 0;
	int32_t head_size_ = 0;
};
}  // namespace op


namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
					const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
					const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
						   const tensor::Tensor& cos_cache);

}  // namespace kernel

#endif //ROPE_CUH

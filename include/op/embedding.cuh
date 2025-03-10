//
// Created by CSWH on 2024/11/19.
//
# include "layer.hpp"

#ifndef EMBEDDING_CUH
#define EMBEDDING_CUH
namespace op {
struct EmbeddingOutput {
	tensor::Tensor input_tokens;
	tensor::Tensor input_embeddings;
	tensor::Tensor input_token_num;
	explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
							 tensor::Tensor input_token_num)
		: input_tokens(std::move(input_tokens)),
		  input_embeddings(std::move(input_embeddings)),
		  input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
public:
	explicit EmbeddingLayer(int32_t dim, int32_t seq_len,
							int32_t vocab_size);

	bool forward() override;

	using Layer::forward;

private:
	int32_t dim_ = 0;
	int32_t seq_len_ = 0;
	int32_t vocab_size_ = 0;
};
}  // namespace op

namespace kernel {
void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
				   const tensor::Tensor& output, int32_t vocab_size);
}
#endif //EMBEDDING_CUH

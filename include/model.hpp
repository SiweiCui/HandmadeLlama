//
// Created by CSWH on 2024/11/18.
//

#ifndef MODEL_HPP
#define MODEL_HPP
#include "embedding.cuh"
# include "base.hpp"
# include "row_model_data.hpp"
# include "tensor.hpp"
# include "sampler.cuh"
# include "encode.hpp"
# include <string>
# include <map>

namespace model {
// 只支持量化模型和CUDA
class Model {
public:
	// 数据读入以及必要的初始化
	int32_t group_size_ = 1;
	std::unique_ptr<base::TransformerConfig> config_ = std::make_unique<base::TransformerConfig>();
	std::string token_path_;
	std::string model_path_;
	std::unique_ptr<op::EncodeLayerBase> encode_layer_;
	std::map<base::ModelBufferType, tensor::Tensor> buffers_; // 激活值等临时变量存储, 内存复用
	std::unique_ptr<sampler::Sampler> sampler_;
	std::shared_ptr<RawModelData> raw_model_data_;
	base::AttentionConfig attention_config_ = base::AttentionConfig::kFlashAttention;
	base::SamplerConfig sampler_config_ = base::SamplerConfig::kTopkSampler;
	base::DeviceType device_type_ = base::DeviceType::kDeviceCUDA;
	base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
	base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;

public:
	explicit Model(base::TokenizerType tokenizer_type,
	               base::ModelType model_type,
	               std::string token_path,
	               std::string model_path,
	               base::AttentionConfig attention_config, base::SamplerConfig sampler_config);
	// 类相关
	virtual bool init() = 0;
	virtual bool init(size_t k) = 0;
	virtual void init_mem() = 0;
	virtual bool create_layers() = 0;
	virtual void create_nonparam_layers() = 0;
	virtual void create_param_quant_layers() = 0;
	virtual bool create_encode_layer();

	// 模型相关
	virtual bool predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
						   bool is_prompt, int& next) const = 0;
	virtual bool forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
		     int& next) const = 0;
	virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
															   int32_t token_pos) const;
	virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

	// 模型buffer
	virtual tensor::Tensor& get_buffer(base::ModelBufferType buffer_idx);
	virtual const tensor::Tensor& get_buffer(base::ModelBufferType buffer_idx) const;
	virtual bool insert_buffer(base::ModelBufferType buffer_idx, const tensor::Tensor& tensor);

	// 处理句子
	virtual bool is_sentence_ending(int32_t token_idx) const;
	virtual std::vector<int32_t> encode(const std::string& sentence) const;
	virtual std::string decode(int32_t token_idx) const;
	virtual std::string decode(std::vector<int32_t> token_idxs) const;
	virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
	                                  const op::EmbeddingOutput& embedding_output,
	                                  bool is_prompt) const;
	virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

	// 导入数据和模型信息
	virtual bool read_model_file();
	virtual bool generate_model_infos(const base::ModelConfig& config);
	virtual bool gen_model_from_file();

};
}


#endif //MODEL_HPP

//
// Created by CSWH on 2024/11/18.
//

#ifndef MODEL_HPP
#define MODEL_HPP
# include "base.hpp"
# include "row_model_data.hpp"
# include "tensor.hpp"
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
		std::map<base::ModelBufferType, tensor::Tensor> buffers_;
		std::shared_ptr<RawModelData> raw_model_data_;
		base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
		base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
		base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;

	public:
		explicit Model(base::TokenizerType tokenizer_type,
		               base::ModelType model_type,
		               std::string token_path,
		               std::string model_path);
		// 我们只实现GPU版本
		virtual bool init();

	//protected:
		virtual bool read_model_file();
		virtual bool generate_model_infos(const base::ModelConfig& config);

	};
}


#endif //MODEL_HPP

//
// Created by CSWH on 2024/11/18.
//
# include "model.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
namespace model {
	Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path, std::string model_path)
		: tokenizer_type_(tokenizer_type),
		model_type_(model_type),
		token_path_(std::move(token_path)),
		model_path_(std::move(model_path)){}

	bool Model::read_model_file() {

		// 读取config并进行配置
		FILE* file = fopen(model_path_.data(), "rb");
		if (!file) {LOG(FATAL) << "Failed to open model file " << model_path_ << "\n";}
		auto config = base::ModelConfig{}; // 聚合初始化
		fread(&config, sizeof(base::ModelConfig), 1, file);
		auto gen_status = generate_model_infos(config);
		CHECK(gen_status == true);

		// 读入量化的group_size
		fread(&group_size_, sizeof(int32_t), 1, file);


		// 设置权重指针
		raw_model_data_ = std::make_shared<RawModelDataInt8>();
		int fd = open(model_path_.data(), O_RDONLY);
		struct stat sb;
		fstat(fd, &sb);
		raw_model_data_->file_size = sb.st_size;
		raw_model_data_->fd = fd;
		raw_model_data_->data = mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
		raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + sizeof(base::ModelConfig) + sizeof(group_size_);

		CHECK_NE(raw_model_data_, nullptr);
		return true;
	}

	bool Model::generate_model_infos(const base::ModelConfig& config) {
		config_->dim_ = config.dim;
		config_->hidden_dim_ = config.hidden_dim;
		config_->layer_num_ = config.layer_num;
		config_->head_num_ = config.head_num;
		config_->kv_head_num_ = config.kv_head_num;
		config_->seq_len_ = config.seq_len;

		config_->kv_mul_ = config.head_num / config.kv_head_num;
		config_->head_size_ = config.dim / config.head_num;
		// 对KV进行线性变换的矩阵维度
		config_->kv_dim_ = config_->head_size_ * config.kv_head_num;

		if (config.vocab_size > 0) {
			config_->is_shared_weight_ = true;
		} else {
			config_->is_shared_weight_ = false;
		}

		config_->vocab_size_ = std::abs(config.vocab_size);
		return true;
	}

	bool Model::init() {
		LOG(ERROR) << "Not implemented yet";
		return false;
	}

}


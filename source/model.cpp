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

	printf("layer_num = %d\n", config_->layer_num_);
	printf("seq_len = %d\n", config_->seq_len_);
	printf("head_num = %d\n", config_->head_num_);
	printf("kv_head_num = %d\n", config_->kv_head_num_);
	printf("kv_mul = %d\n", config_->kv_mul_);
	printf("head_size = %d\n", config_->head_size_);
	printf("kv_dim = %d\n", config_->kv_dim_);

	return true;
}

// 内存复用
bool Model::insert_buffer(base::ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
	if (buffers_.count(buffer_idx) > 0) {
		return false;
	}
	if (tensor.is_empty()) {
		return false;
	}
	buffers_.insert({buffer_idx, tensor});
	return true;
}

tensor::Tensor& Model::get_buffer(base::ModelBufferType buffer_idx) {
	CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
	return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(base::ModelBufferType buffer_idx) const {
	CHECK_GT(buffers_.count(buffer_idx), 0);
	return buffers_.at(buffer_idx);
}

bool Model::create_encode_layer() {
	using namespace base;

	// create token encode decode layer
	if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
		encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
	}
	if (!encode_layer_) {
		return false;
	}

	config_->vocab_size_ = encode_layer_->vocab_size();
	if (config_->vocab_size_ <= 0) {
		return false;
	}
	return true;
}

bool Model::gen_model_from_file() {
	using namespace base;
	config_ = std::make_unique<TransformerConfig>();

	// init sentence piece processor
	// google sentence piece
	auto create_encode_status = create_encode_layer();
	if (!create_encode_status) {
	LOG(ERROR) << "Create the encode layer failed!";
	return create_encode_status;
	}
	// mmap
	auto mmap_status = read_model_file();
	if (!mmap_status) {
	LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
	return mmap_status;
	}
	auto layer_create_status = create_layers();
	if (!layer_create_status) {
	LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
	return layer_create_status;
	}

	return true;
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
	CHECK(encode_layer_ != nullptr);
	return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
	CHECK(this->encode_layer_ != nullptr);
	return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
	CHECK(this->encode_layer_ != nullptr);
	return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
	CHECK(this->encode_layer_ != nullptr);
	return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
	int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
	int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

	float* key_cache_ptr =
	  const_cast<float*>(get_buffer(base::ModelBufferType::kKeyCache).ptr<float>(cache_offset));
	float* val_cache_ptr =
	  const_cast<float*>(get_buffer(base::ModelBufferType::kValueCache).ptr<float>(cache_offset));

	tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
	                 key_cache_ptr);
	tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
	                 val_cache_ptr);
	key.set_device_type(device_type_);
	val.set_device_type(device_type_);
	return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                       const op::EmbeddingOutput& embedding_output,
                                       bool is_prompt) const {
	const int32_t pos = pos_tensor.index<int32_t>(0);
	auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

	int32_t index = 0;
	if (is_prompt) {
	index = pos;
	}
	std::shared_ptr<base::Buffer> input_emb_buffer =
	  std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
	                                 input_embeddings.ptr<float>(index * config_->dim_), true);

	tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
	input.assign(input_emb_buffer);
	input.set_device_type(device_type_);
	return input;
}
}


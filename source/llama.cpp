//
// Created by CSWH on 2024/11/21.
//
# include "llama.hpp"

#include <add.cuh>
#include <matmul.cuh>
#include <mha.cuh>
#include <rmsnorm.cuh>
#include <rope.cuh>
#include <softmax.cuh>
#include <swiglu.cuh>

#include <glog/logging.h>


namespace model {
void LLama2Layers::to_cuda() {
    if (add_layer_) {
        add_layer_->to_cuda();
    }

    if (rope_layer_) {
        rope_layer_->to_cuda();
    }

    if (swiglu_layer_) {
        swiglu_layer_->to_cuda();
    }

    if (cls_layer_) {
        cls_layer_->to_cuda();
    }

    if (embedding_layer_) {
        embedding_layer_->to_cuda();
    }

    if (mha_layer_) {
        mha_layer_->to_cuda();
    }

    for (auto& weight_layer : wq_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wk_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wv_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : wo_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w1_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w2_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& weight_layer : w3_layers_) {
        if (weight_layer) {
            weight_layer->to_cuda();
        }
    }

    for (auto& rms_norm_layer : rmsnorm_layers_) {
        if (rms_norm_layer) {
            rms_norm_layer->to_cuda();
        }
    }
}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                     std::string model_path)
: Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
        std::move(model_path), base::AttentionConfig::kFlashAttention, base::SamplerConfig::kTopkSampler) {}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                     std::string model_path, base::AttentionConfig attention_config, base::SamplerConfig sampler_config)
: Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
        std::move(model_path), attention_config, sampler_config) {}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                     std::string model_path, base::SamplerConfig sampler_config)
: Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
        std::move(model_path), base::AttentionConfig::kFlashAttention, sampler_config) {}

bool LLama2Model::init() {
    using namespace base;
    if (token_path_.empty()) {
        return false;
    }

    bool read_status = gen_model_from_file();
    if (!read_status) {
        return read_status;
    }
    init_mem();
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache));

    if (sampler_config_ == base::SamplerConfig::kTopkSampler) {
        sampler_ = std::make_unique<sampler::TopKSampler>(3);
    }else {
        sampler_ = std::make_unique<sampler::ArgmaxSampler>();
    }
    return true;
}

bool LLama2Model::init(size_t topk) {
    using namespace base;
    if (token_path_.empty()) {
        return false;
    }

    bool read_status = gen_model_from_file();
    if (!read_status) {
        return read_status;
    }

    init_mem();

    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache));

    if (sampler_config_ == base::SamplerConfig::kTopkSampler) {
        sampler_ = std::make_unique<sampler::TopKSampler>(topk);
    }else {
        sampler_ = std::make_unique<sampler::ArgmaxSampler>();
    }
    return true;
}

bool LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                              int& next) const {
    if (input.is_empty()) {
        return false;
    }
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        return false;
    }

    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        attention_rms(layer_idx, input);

        // attention (wq wk wv @ input)
        attention_qkv(layer_idx, pos_tensor);

        // multi-head attention
        attention_mha(layer_idx, pos_tensor);

        // feed forward
        feed_forward(layer_idx, input);
    }
    cls_logits(input);
    return true;
}

bool LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                              bool is_prompt, int& next) const {
    auto status = forward(input, pos_tensor, next);
    if (!status) {
        return status;
    }
    next = post_processing(pos_tensor, is_prompt);
    return true;
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
    tensor::Tensor forward_output = get_buffer(base::ModelBufferType::kForwardOutput);
    const float* forward_logits = forward_output.ptr<float>(); // 这里实际上不是softmax的输出
    int32_t next = 0;
    if (is_prompt) {
        next = -1;
    } else {
        if(sampler_config_ == base::SamplerConfig::kTopkSampler) {
            tensor::Tensor forward_logits_softmax = get_buffer(base::ModelBufferType::kForwardOutputSoftmax);
            llama_layers_->final_softmax_layer_->forward(forward_output, forward_logits_softmax);
            next = static_cast<int32_t>(sampler_->sample(forward_logits_softmax.ptr<float>(), forward_logits_softmax.size()));
        }else if(sampler_config_ == base::SamplerConfig::kArgMaxSampler) {
            // 如果是argmax, 不用再softmax了.
            next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size()));
        }
    }
    return next;
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK(llama_layers_ != nullptr);
    // attn rmsnorm
    tensor::Tensor rmsnorm_output = get_buffer(base::ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
    if (!rmsnorm_layer) {
        LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
    }
    rmsnorm_layer->forward(input, rmsnorm_output);
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(llama_layers_ != nullptr);
    // 拿出q, k, v, 都是当前时间点的向量
    // query buffer
    tensor::Tensor query = this->get_buffer(base::ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);
    // kv cache
    const auto& [key, val] = slice_kv_cache(layer_idx, pos);

    // q的线性变换
    const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
    auto rmsnorm_output = get_buffer(base::ModelBufferType::kOutputRMSNorm);
    query_layer->forward(rmsnorm_output, query);

    // key的线性变换
    const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
    key_layer->forward(rmsnorm_output, key);

    // value的线性变换
    const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    value_layer->forward(rmsnorm_output, val);

    // rope
    CHECK_NE(llama_layers_->rope_layer_, nullptr)
        << "The RoPE layer in the attention block is null pointer.";
    (llama_layers_->rope_layer_->forward(
        query, key, pos_tensor, get_buffer(base::ModelBufferType::kSinCache),
        get_buffer(base::ModelBufferType::kCosCache), tensor::Tensor{}));
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(llama_layers_ != nullptr);
    tensor::Tensor key_cache = get_buffer(base::ModelBufferType::kKeyCache);
    tensor::Tensor val_cache = get_buffer(base::ModelBufferType::kValueCache);

    tensor::Tensor mha_output = get_buffer(base::ModelBufferType::kOutputMHA);
    tensor::Tensor score_storage = get_buffer(base::ModelBufferType::kScoreStorage);
    tensor::Tensor query = this->get_buffer(base::ModelBufferType::kQuery);

    const auto& mha_layer = llama_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    int pos = pos_tensor.index<int32_t>(0);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos); // 共享指针安全向下转型
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output);

    // wo @ attention output
    tensor::Tensor attn_output = get_buffer(base::ModelBufferType::kAttnOutput);
    const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
    wo_layer->forward(mha_output, attn_output);
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
      CHECK(llama_layers_ != nullptr);
      // residual add
      CHECK_NE(llama_layers_->add_layer_, nullptr)
          << "The add layer in the feedforward block is null pointer";
      llama_layers_->add_layer_->forward(input, get_buffer(base::ModelBufferType::kAttnOutput), input);

      // ffn rmsnorm
      tensor::Tensor ffn_norm_output = get_buffer(base::ModelBufferType::kFFNRMSNorm);
      const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
      CHECK_NE(ffn_rmsnorm, nullptr)
          << "The final rmsnorm layer in the feedforward block is null pointer";
      ffn_rmsnorm->forward(input, ffn_norm_output);

      // w1
      tensor::Tensor w1_output = get_buffer(base::ModelBufferType::kW1Output);
      const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
      CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
      w1_layer->forward(ffn_norm_output, w1_output);

      // w3
      tensor::Tensor w3_ouput = get_buffer(base::ModelBufferType::kW3Output);
      const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
      CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
      w3_layer->forward(ffn_norm_output, w3_ouput);

      // SwiGLU
      CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
          << "The swiglu layer in the feedforward block is null pointer";
      llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output);

      // w2
      tensor::Tensor w2_output = get_buffer(base::ModelBufferType::kW2Output);
      const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
      CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
      w2_layer->forward(w1_output, w2_output);

      // residual add
      CHECK_NE(llama_layers_->add_layer_, nullptr)
          << "The add layer in the feedforward block is null pointer";
      (llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) const {
    CHECK(llama_layers_ != nullptr);
    const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm, nullptr);
    norm->forward(input, input);

    tensor::Tensor forward_output = get_buffer(base::ModelBufferType::kForwardOutput);
    CHECK_NE(llama_layers_->cls_layer_, nullptr);
    (llama_layers_->cls_layer_->forward(input, forward_output));

}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
    auto input_tokens = get_buffer(base::ModelBufferType::kInputTokens);
    auto input_embeddings = get_buffer(base::ModelBufferType::kInputEmbeddings);
    if (input_tokens.size() != tokens.size()) { // 调整大小, 涉及到buffer的重新创建
        // prompt阶段, 调整成prompt_len行, decoding阶段, 调整回1行, 并一直是一行
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
    }
    for (int32_t i = 0; i < tokens.size(); ++i) {
        input_tokens.index<int32_t>(i) = tokens.at(i);
    }

    auto input_token_num =
        tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
    LOG_IF(FATAL, !llama_layers_->embedding_layer_)
        << "The embedding layer in the llama2 model is null pointer.";

    llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings);

    op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
    return output;
}

bool LLama2Model::create_layers() {
    using namespace base;
    if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
    }

    create_param_quant_layers();
    create_nonparam_layers();

    if (!llama_layers_->embedding_layer_) {
        return false;
    }

    return true;
}

void LLama2Model::create_param_quant_layers() {
    CHECK(llama_layers_ != nullptr);

    size_t pos = 0;
    int32_t dim = config_->dim_;

    // query
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(dim, dim);
        wq->set_group_size(group_size_);
        //is_quant默认为true
        wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));

        llama_layers_->wq_layers_.push_back(wq);
        pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
    }

    // key
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(config_->kv_dim_, dim);
        wk->set_group_size(group_size_);
        wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos));
        llama_layers_->wk_layers_.push_back(wk);
        pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
    }

    // value
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(config_->kv_dim_, dim);
        wv->set_group_size(group_size_);
        wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos));
        llama_layers_->wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
    }

    // output
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(dim, dim);
        wo->set_group_size(group_size_);
        wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
        llama_layers_->wo_layers_.push_back(wo);
        pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
    }

    // w1 layers
    int32_t hidden_dim = config_->hidden_dim_;
        for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(hidden_dim, dim);
        w1->set_group_size(group_size_);
        w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
        llama_layers_->w1_layers_.push_back(w1);
        pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
    }

    // w2 layers
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(dim, hidden_dim);
        w2->set_group_size(group_size_);
        w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos));
        llama_layers_->w2_layers_.push_back(w2);
        pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
    }

    // w3 layers
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(hidden_dim, dim);
        w3->set_group_size(group_size_);
        w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
        llama_layers_->w3_layers_.push_back(w3);
        pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
    }

    // wcls layer
    auto cls_layer = std::make_shared<op::MatmulLayer>(config_->vocab_size_, dim);
    cls_layer->set_group_size(group_size_);
    if (config_->is_shared_weight_) {
        // using token embedding weight
        cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos));
    } else {
        // no shared
        cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos));
        pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
    }
    llama_layers_->cls_layer_ = cls_layer;

    // embedding layer
    float* weight_ptr = (float*)raw_model_data_->weight(pos);
    llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
    llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              false);
    weight_ptr += config_->vocab_size_ * dim;

    // rmsnorm attention attention,ffn,final
    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, false);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
    }
}

void LLama2Model::create_nonparam_layers() {
    CHECK(llama_layers_ != nullptr);
    llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
        config_->dim_, config_->kv_dim_, config_->head_size_);

    llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
        config_->head_size_, attention_config_);

    llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>();

    llama_layers_->swiglu_layer_ =
        std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);

    llama_layers_->final_softmax_layer_ = std::make_shared<op::SoftmaxLayer>(device_type_);
}

void LLama2Model::init_mem() {

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        llama_layers_->to_cuda();
    }

    std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    // input_tokens在cpu, 初始大小设置为1, 如果后续需要变动(如prompt阶段), 在embedding()中通过tensor的reshape()完成
    tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    // 其他的都在gpu
    tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc_cu); // 初始行数同样为1
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc_cu);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc_cu);

    CHECK(insert_buffer(base::ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(base::ModelBufferType::kCosCache, cos_cache));

    CHECK(insert_buffer(base::ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(base::ModelBufferType::kInputEmbeddings, input_embeddings));

    tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc_cu);
    CHECK(insert_buffer(base::ModelBufferType::kOutputRMSNorm, rms_output)); // 内存复用
    CHECK(insert_buffer(base::ModelBufferType::kOutputMHA, rms_output));
    CHECK(insert_buffer(base::ModelBufferType::kW2Output, rms_output));
    CHECK(insert_buffer(base::ModelBufferType::kFFNRMSNorm, rms_output));

    tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc_cu);
    tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc_cu);

    CHECK(insert_buffer(base::ModelBufferType::kW1Output, w1_output));
    CHECK(insert_buffer(base::ModelBufferType::kW3Output, w3_output));

    // kv cache
    tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc_cu);
    tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc_cu);

    CHECK(insert_buffer(base::ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(base::ModelBufferType::kValueCache, value_cache));

    // Wq query output
    tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc_cu);
    CHECK(insert_buffer(base::ModelBufferType::kQuery, query));

    // Pos tensor
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_buffer(base::ModelBufferType::kInputPos, pos_tensor));

    // Attention output
    tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc_cu);
    CHECK(insert_buffer(base::ModelBufferType::kScoreStorage, attn));
    CHECK(insert_buffer(base::ModelBufferType::kAttnOutput, query)); // 注意力的输出复用query

    // final forward output
    tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc_cu);
    CHECK(insert_buffer(base::ModelBufferType::kForwardOutput, forward_output));

    // final softmax output
    if (sampler_config_ == base::SamplerConfig::kTopkSampler) {
        tensor::Tensor forward_output_softmax(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc_cu);
        CHECK(insert_buffer(base::ModelBufferType::kForwardOutputSoftmax, forward_output_softmax));
    }
}
}

//
// Created by CSWH on 2024/11/21.
//

#ifndef LLAMA_HPP
#define LLAMA_HPP
# include "model.hpp"

namespace model {
struct LLama2Layers {
    std::shared_ptr<op::Layer> add_layer_;
    std::shared_ptr<op::Layer> rope_layer_;
    std::shared_ptr<op::Layer> swiglu_layer_;
    std::shared_ptr<op::Layer> mha_layer_;

    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;

    std::vector<std::shared_ptr<op::Layer>> w1_layers_;
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;
    std::shared_ptr<op::Layer> cls_layer_;
    std::shared_ptr<op::Layer> final_softmax_layer_;

    std::shared_ptr<op::Layer> embedding_layer_;

    void to_cuda();
};

class LLama2Model : public Model {
    std::unique_ptr<LLama2Layers> llama_layers_;

public:
    explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path);

    LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                    std::string model_path, base::AttentionConfig attention_config, base::SamplerConfig sampler_config);

    LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                    std::string model_path, base::SamplerConfig sampler_config);

    bool init() override;
    bool init(size_t topk) override;

    // 模型相关
    bool predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;
    bool forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;
    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

private:
    // 类相关
    void init_mem() override;
    bool create_layers() override;
    void create_nonparam_layers() override;
    void create_param_quant_layers() override;

    // 计算相关
    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
    void cls_logits(const tensor::Tensor& input) const;

    // 字符处理相关
    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

};
}
#endif //LLAMA_HPP

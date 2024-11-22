//
// Created by CSWH on 2024/11/18.
//

/*
# include <model.hpp>
TEST(test_model, test_mdoel_load) {
	model::Model model(base::TokenizerType::kEncodeSpe,
						base::ModelType::kModelTypeLLama2,
						"/home/csw/big_model/lession_model/tokenizer.model", // 暂时还没用到
						"/home/csw/big_model/lession_model/tinyllama_int8.bin"
						);
	model.read_model_file();
	printf("d_model: %d\n", model.config_->dim_);
	printf("head_num: %d\n", model.config_->head_num_);
	printf("head_size: %d\n", model.config_->head_size_);
	printf("hidden_dim: %d\n", model.config_->hidden_dim_);
	printf("is_shared_weight: %d\n", model.config_->is_shared_weight_);
	printf("kv_head_num: %d\n", model.config_->kv_head_num_);
	printf("layer_num: %d\n", model.config_->layer_num_);
	printf("seq_len: %d\n", model.config_->seq_len_);
	printf("vocab_size: %d\n", model.config_->vocab_size_);
}
/
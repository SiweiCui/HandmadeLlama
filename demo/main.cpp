//
// Created by CSWH on 2024/11/17.
//
#include <stdio.h>
#include "llama.hpp"
#include <glog/logging.h>

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(base::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        if (model.is_sentence_ending(next)) {
            break;
        }
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        } else {
            words.push_back(next);
        }
        pos += 1;
    }
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path,
        base::AttentionConfig::kFlashAttention, base::SamplerConfig::kTopkSampler);
    auto init_status = model.init(3);
    if (!init_status) {
        LOG(FATAL) << "The model init failed";
    }
    // const std::string& sentence = "Hello, what's up bro?";
    // const std::string& sentence = "How's the weather today?\n";
    const std::string& sentence = "How to define probability mathematically? Answer this question.\n";
    // const std::string& sentence = "How many 'r' in 'strawberry'?\n";
    // const std::string& sentence = "Which is bigger, 9.11 or 9.8? Calculate 9.11 minus 9.8, and answer the question.\n";
    // const std::string& sentence = "Write the code of quick sort algorithm in C++\n";


    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, sentence, 256, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
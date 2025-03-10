//
// Created by CSWH on 2024/11/19.
//

#ifndef SAMPLER_CUH
#define SAMPLER_CUH
#include <cstddef>
#include "base.hpp"
#include<random>

namespace sampler {
class Sampler {
public:
	virtual size_t sample(const float* logits, size_t size) = 0;
};

class ArgmaxSampler : public Sampler {
public:
	size_t sample(const float* logits, size_t size) override;
};

class TopKSampler : public Sampler {
	size_t k_;
	// 随机数引擎
	std::mt19937 gen;
	// 均匀分布对象，范围为0到1
	std::uniform_real_distribution<float> dis;
public:
	explicit TopKSampler(size_t k);
	size_t sample(const float* logits, size_t size) override;
};
}  // namespace sampler

namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size);
size_t topk_kernel_cu(const float* input_ptr, size_t size, size_t K, float rand_num);
}


#endif //SAMPLER_CUH

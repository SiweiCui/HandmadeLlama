//
// Created by CSWH on 2024/11/19.
//

#ifndef SAMPLER_CUH
#define SAMPLER_CUH
#include <cstddef>
#include "base.hpp"

namespace sampler {
	class Sampler {
	public:
		virtual size_t sample(const float* logits, size_t size) = 0;

	};

	class ArgmaxSampler : public Sampler {
	public:
		size_t sample(const float* logits, size_t size) override;
	};
}  // namespace sampler

namespace kernel {
	size_t argmax_kernel_cu(const float* input_ptr, size_t size);
}


#endif //SAMPLER_CUH

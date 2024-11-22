//
// Created by CSWH on 2024/11/19.
//

#ifndef SWIGLU_CUH
#define SWIGLU_CUH
#include "layer.hpp"
namespace op {
	class SwiGLULayer : public Layer {
	public:
		explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

		bool forward() override;

	private:
		int32_t hidden_dim_ = 0;
	};
}  // namespace op

namespace kernel {
	void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
						  const tensor::Tensor& output);
}

#endif //SWIGLU_CUH

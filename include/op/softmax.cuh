//
// Created by CSWH on 2025/3/10.
//
#include"layer.hpp"

#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH
namespace op {
class SoftmaxLayer : public Layer {
public:
	explicit SoftmaxLayer(base::DeviceType device_type);

	bool forward() override;
	using Layer::forward;

};
}  // namespace op

namespace kernel {
void softmax_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& output);
}

#endif //SOFTMAX_CUH

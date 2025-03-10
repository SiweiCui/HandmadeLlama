//
// Created by CSWH on 2024/11/19.
//

#ifndef ADD_CUH
#define ADD_CUH

# include "layer.hpp"
# include "tensor.hpp"

namespace op {
class VecAddLayer : public Layer {
public:
	explicit VecAddLayer();

	bool forward() override;

	using Layer::forward;
};
}


namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
			   const tensor::Tensor& output, void* stream = nullptr);
}

#endif //ADD_CUH

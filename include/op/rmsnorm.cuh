//
// Created by CSWH on 2024/11/19.
//

#ifndef RMSNORM_CUH
#define RMSNORM_CUH

#include "layer.hpp"
#include "tensor.hpp"
namespace op {
class RmsNormLayer : public LayerParam {
public:
	explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

	bool forward() override;
	using Layer::forward;

private:
	int32_t dim_ = 0;
};
}  // namespace op

namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
					   const tensor::Tensor& output, void* stream = nullptr);
}

#endif //RMSNORM_CUH

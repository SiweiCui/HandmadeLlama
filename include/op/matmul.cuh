//
// Created by CSWH on 2024/11/19.
//

#ifndef MATMUL_CUH
#define MATMUL_CUH
# include "layer.hpp"

namespace op {
	class MatmulLayer : public LayerParam {
	private:
		int32_t dim0_ = 0;
		int32_t dim1_ = 0;
		bool has_bias_ = false;
		std::vector<tensor::Tensor> bias_;
	public:
		explicit MatmulLayer(int32_t dim0, int32_t dim1, bool has_bias = false);

		bool forward() override;

		bool set_bias(int32_t idx, int32_t& dim, const void* bias_ptr);

		tensor::Tensor& get_bias(int32_t idx);
		const tensor::Tensor& get_bias(int32_t idx) const;

		void to_cuda() override;

	};
}

namespace kernel {
	void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
							const tensor::Tensor& output, int32_t group_size,
							const tensor::Tensor& scale);
}


#endif //MATMUL_CUH

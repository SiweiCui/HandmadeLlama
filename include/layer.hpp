//
// Created by CSWH on 2024/11/18.
//

#ifndef LAYER_HPP
#define LAYER_HPP
# include "base.hpp"
# include "tensor.hpp"
# include <string>
namespace op {
class BaseLayer {
protected:
	std::string layer_name_;
	base::LayerType layer_type_ = base::LayerType::kLayerUnknown;
	// 仅支持int8量化
	base::DataType data_type_ = base::DataType::kDataTypeInt8;
	// 初始化都在CPU上, 移动到GPU上只能靠to_cuda()
	base::DeviceType device_type_ = base::DeviceType::kDeviceCPU;

public:
	explicit BaseLayer(base::LayerType layer_type,
		base::DataType data_type,
		std::string layer_name = "");

	base::DataType data_type() const;
	base::LayerType layer_type() const;

	// 层初始化方法
	virtual bool init() = 0;

	// 最多支持5个输入
	virtual bool forward() = 0;
	virtual bool forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;
	virtual bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
								 const tensor::Tensor& output1) = 0;
	virtual bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
								 const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;
	virtual bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
								 const tensor::Tensor& input3, const tensor::Tensor& input4,
								 const tensor::Tensor& output1) = 0;
	virtual bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
								 const tensor::Tensor& input3, const tensor::Tensor& input4,
								 const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

	// 手动设置输入输出
	virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;
	virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;
	virtual size_t input_size() const = 0;
	virtual size_t output_size() const = 0;
	// 获取输入输出
	virtual tensor::Tensor& get_input(int32_t idx) = 0;
	virtual tensor::Tensor& get_output(int32_t idx) = 0;
	virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
	virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

	// 带参情况下, 设置权重
	virtual bool set_weight(int32_t idx, const tensor::Tensor& weight);
	virtual bool set_weight(int32_t idx, const std::vector<int32_t>& dims,
									const void* weight_ptr, bool is_quant);


	// layer name
	const std::string& get_layer_name() const;
	void set_layer_name(const std::string& layer_name);

};

// 不带参数派生类
class Layer : public BaseLayer {
protected:
	std::vector<tensor::Tensor> inputs_;
	std::vector<tensor::Tensor> outputs_;
public:
	explicit Layer(base::LayerType layer_type, std::string layer_name = "");

	bool init() override;

	bool forward() override;
	bool forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;
	bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
	                     const tensor::Tensor& output1) override;
	bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
	                     const tensor::Tensor& input3, const tensor::Tensor& output1) override;
	bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
	                     const tensor::Tensor& input3, const tensor::Tensor& input4,
	                     const tensor::Tensor& output1) override;
	bool forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
	                     const tensor::Tensor& input3, const tensor::Tensor& input4,
	                     const tensor::Tensor& input5, const tensor::Tensor& output1) override;
	void set_input(int32_t idx, const tensor::Tensor& input) override;
	void set_output(int32_t idx, const tensor::Tensor& output) override;
	const tensor::Tensor& get_input(int32_t idx) const override;
	const tensor::Tensor& get_output(int32_t idx) const override;
	tensor::Tensor& get_input(int32_t idx) override;
	tensor::Tensor& get_output(int32_t idx) override;

	size_t input_size() const override;
	size_t output_size() const override;

	void reset_input_size(size_t size);
	void reset_output_size(size_t size);

	virtual void to_cuda();

};

// 带参数派生类, 特指权重层
class LayerParam : public Layer {
public:
	int group_size_ = 0;
	tensor::Tensor scales_;
	std::vector<tensor::Tensor> weights_;

public:
	explicit LayerParam(base::LayerType layer_type,
		std::string layer_name = "");

	size_t weight_size() const;
	void reset_weight_size(size_t size);
	tensor::Tensor& get_weight(int32_t idx);
	const tensor::Tensor& get_weight(int32_t idx) const;

	void to_cuda() override;

	bool set_weight(int32_t idx, const tensor::Tensor& weight) override;
	bool set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, bool is_quant = true) override;
	void set_scales(const tensor::Tensor& scales);
	void set_group_size(int32_t group_size);

	int32_t get_scale_num() const;

	int32_t get_group_size() const;

};


}
#endif //LAYER_HPP

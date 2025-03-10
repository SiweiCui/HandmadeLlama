//
// Created by CSWH on 2024/11/18.
//
# include "layer.hpp"

#include <numeric>

namespace op {
BaseLayer::BaseLayer(base::LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
: layer_type_(layer_type),
  data_type_(data_type),
  layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }
base::LayerType BaseLayer::layer_type() const { return layer_type_; }

// 非纯虚, 必须实现
bool BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
	LOG(ERROR) << "Not implemented yet";
	return false;
}
bool BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
								   const void* weight_ptr, bool is_quant) {
	LOG(ERROR) << "Not implemented yet";
	return false;
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }
void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }


Layer::Layer(base::LayerType layer_type, std::string layer_name)
: BaseLayer(layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

bool Layer::init() { return true; }

bool Layer::forward() {
	LOG(ERROR) << "Not implemented yet";
	return false;
}

bool Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
	this->set_input(0, input1);
	this->set_output(0, output1);
	return this->forward();
}
bool Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
						  const tensor::Tensor& output1) {
	this->set_input(0, input1);
	this->set_input(1, input2);

	this->set_output(0, output1);
	return this->forward();
}
bool Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
						  const tensor::Tensor& input3, const tensor::Tensor& output1) {
	this->set_input(0, input1);
	this->set_input(1, input2);
	this->set_input(2, input3);

	this->set_output(0, output1);
	return this->forward();
}
bool Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
						  const tensor::Tensor& input3, const tensor::Tensor& input4,
						  const tensor::Tensor& output1) {
	this->set_input(0, input1);
	this->set_input(1, input2);
	this->set_input(2, input3);
	this->set_input(3, input4);

	this->set_output(0, output1);
	return this->forward();
}
bool Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
						  const tensor::Tensor& input3, const tensor::Tensor& input4,
						  const tensor::Tensor& input5, const tensor::Tensor& output1) {
	this->set_input(0, input1);
	this->set_input(1, input2);
	this->set_input(2, input3);
	this->set_input(3, input4);
	this->set_input(4, input5);

	this->set_output(0, output1);
	return this->forward();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, inputs_.size());
	this->inputs_.at(idx) = input;
}
void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, outputs_.size());
	this->outputs_.at(idx) = output;
}
const tensor::Tensor& Layer::get_input(int32_t idx) const {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, inputs_.size());
	return inputs_.at(idx);
}
tensor::Tensor& Layer::get_input(int32_t idx) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, inputs_.size());
	return inputs_.at(idx);
}
tensor::Tensor& Layer::get_output(int32_t idx) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, outputs_.size());
	return outputs_.at(idx);
}
const tensor::Tensor& Layer::get_output(int32_t idx) const {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, outputs_.size());
	return outputs_.at(idx);
}

// 先要重新设置大小再添加输入输出, 因为初始vector大小为0.
void Layer::reset_input_size(size_t size) { inputs_.resize(size); }
void Layer::reset_output_size(size_t size) { outputs_.resize(size); }
size_t Layer::input_size() const { return inputs_.size(); }
size_t Layer::output_size() const { return outputs_.size(); }

// 将Layer中的tensor全部转移到cuda上
void Layer::to_cuda() {
	for (auto& input : inputs_) {
		if (!input.is_empty()) {
			input.to_cuda();
		}
	}
	for (auto& output : outputs_) {
		if (!output.is_empty()) {
			output.to_cuda();
		}
	}
}

LayerParam::LayerParam(base::LayerType layer_type, std::string layer_name)
	  : Layer(layer_type, std::move(layer_name)) {}

void LayerParam::to_cuda() {
	Layer::to_cuda();
	for (auto& weight : weights_) {
		weight.to_cuda();
	}
	if (!scales_.is_empty()) {
		scales_.to_cuda();
	}
}

// 方便测试
bool LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, weights_.size());
	// 只能设置FP32的权重
	CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
	if (!weight.is_empty()) {
	 CHECK(weight.device_type() == device_type_);
	}
	weights_.at(idx) = weight;
	return true;
}

/*
 * 将mmap的数据设置成权重, 暂时还是放在cpu里面
 */
bool LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                  const void* weight_ptr, bool is_quant) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, weights_.size());
	CHECK_NE(weight_ptr, nullptr);

	if (is_quant) {
		// int8量化后的模型, 先设置权重
		size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(int8_t), std::multiplies<>());
		std::shared_ptr<base::Buffer> buffer =
			std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
		buffer->set_device_type(base::DeviceType::kDeviceCPU);
		tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
		weight.set_device_type(base::DeviceType::kDeviceCPU);
		CHECK(weight.assign(buffer));
		weights_.at(idx) = weight;

		// 后设置scale
		const int32_t weight_size = static_cast<int32_t>(weight.size());
		CHECK(weight_size % group_size_ == 0);
		int32_t scale_nums = weight_size / group_size_;
		scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
								 reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
		scales_.set_device_type(base::DeviceType::kDeviceCPU);

	} else {
		size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
		std::shared_ptr<base::Buffer> buffer =
			std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
		buffer->set_device_type(base::DeviceType::kDeviceCPU);
		tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
		weight.set_device_type(base::DeviceType::kDeviceCPU);
		CHECK(weight.assign(buffer));
		weights_.at(idx) = weight;
	}

	return true;
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, weights_.size());
	return weights_.at(idx);
}


void LayerParam::set_scales(const tensor::Tensor& scales) {
	CHECK(!scales.is_empty());
	this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
	CHECK(!scales_.is_empty());
	return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }



tensor::Tensor& LayerParam::get_weight(int32_t idx) {
	CHECK_GE(idx, 0);
	CHECK_LT(idx, weights_.size());
	return weights_.at(idx);
}

}

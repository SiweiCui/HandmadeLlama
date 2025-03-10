//
// Created by CSWH on 2024/11/17.
//

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <glog/logging.h>
#include <memory>
#include <vector>
#include "buffer.hpp"
#include "../../../../usr/local/cuda-11.6/targets/x86_64-linux/include/cuda_runtime_api.h"

namespace tensor{

class Tensor {
private:
    size_t size_;
    std::vector<int32_t> dims_;
    std::shared_ptr<base::Buffer> buffer_;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;

public:
	/*
	 *默认初始化. 方便后续编程.
	 *因为tensor是不持有数据的, 所以我们在类里面使用的tensor都不是指针.
	 *提供默认初始化, 可以避免持有tensor的类的构造中必须初始化tensor
	 */
	explicit Tensor() = default;
    explicit Tensor(base::DataType,
                    int32_t dim0,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr
                    );
    explicit Tensor(base::DataType,
                int32_t dim0,
                    int32_t dim1,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
                );
    explicit Tensor(base::DataType,
                int32_t dim0,
                    int32_t dim1,
                    int32_t dim2,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
                );
    explicit Tensor(base::DataType,
                int32_t dim0,
                    int32_t dim1,
                    int32_t dim2,
                    int32_t dim3,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
                );
    explicit Tensor(base::DataType,
                int32_t dim0,
                    int32_t dim1,
                    int32_t dim2,
                    int32_t dim3,
                    int32_t dim4,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
                );
    explicit Tensor(base::DataType,
                std::vector<int32_t> dims,
                bool need_alloc = false,
                std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                void* ptr = nullptr
                );
    void to_cpu();
    void to_cuda();
    bool is_empty() const;
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DataType data_type,
                     bool need_alloc,
                     void* ptr
                     );

    void reshape(const std::vector<int32_t>& dims);

    std::shared_ptr<base::Buffer> get_buffer() const;

    size_t size() const;

    size_t byte_size() const;

    int32_t dims_size() const;

    base::DataType data_type() const;

    int32_t get_dim(int32_t idx) const;

    const std::vector<int32_t>& dims() const;

    std::vector<size_t> strides() const;

    bool assign(std::shared_ptr<base::Buffer> buffer);

    void reset(base::DataType data_type, const std::vector<int32_t>& dims);

    void set_device_type(base::DeviceType device_type) const;

    base::DeviceType device_type() const;

    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

    template <typename T>
    const T* ptr() const;

    template <typename T>
    T* ptr();

    template <typename T>
    T* ptr(int64_t index);

    template <typename T>
    const T* ptr(int64_t index) const;

    template <typename T>
    T& index(int64_t offset);

    template <typename T>
    const T& index(int64_t offset) const;

    Tensor clone() const;

	template<typename T>
	void show_digits(size_t shows) const;

	template<typename T>
	void show_top5() const;
};

template<typename T>
T& Tensor::index(int64_t offset){
	CHECK_GE(offset, 0);
	CHECK_LT(offset, this->size());
	T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
	return val;
}
template <typename T>
const T& Tensor::index(int64_t offset) const {
	CHECK_GE(offset, 0);
	CHECK_LT(offset, this->size());
	const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
	return val;
}
template <typename T>
const T* Tensor::ptr() const {
	if (!buffer_) {
		return nullptr;
	}
	return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}
template <typename T>
T* Tensor::ptr() {
	if (!buffer_) {
		return nullptr;
	}
	return reinterpret_cast<T*>(buffer_->ptr());
}
template <typename T>
T* Tensor::ptr(int64_t index) {
	CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
		<< "The data area buffer of this tensor is empty or it points to a null pointer.";
	return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}
template <typename T>
const T* Tensor::ptr(int64_t index) const {
	CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
		<< "The data area buffer of this tensor is empty or it points to a null pointer.";
	return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

template<typename T>
void Tensor::show_digits(size_t shows) const{
	std::shared_ptr<base::Buffer> thisBuffer = this->get_buffer();
	T* data_ptr = nullptr;
	bool isMalloc = false;
	if(thisBuffer->device_type() == base::DeviceType::kDeviceCUDA) {
		data_ptr = (T*)malloc(sizeof(T) * this->size());
		isMalloc = true;
		cudaMemcpy(data_ptr, thisBuffer->ptr(), sizeof(T) * this->size(), cudaMemcpyDeviceToHost);
	}else {
		data_ptr = reinterpret_cast<T*>(thisBuffer->ptr());
	}
	for(size_t i = 0; i < shows; i++) {
		printf("%f\t", (float)(data_ptr)[i]);
		fflush(stdout);
	}
	if(isMalloc) {
		free(data_ptr);
	}
}

template<typename T>
void Tensor::show_top5() const {
	std::shared_ptr<base::Buffer> thisBuffer = this->get_buffer();
	T* data_ptr = nullptr;
	bool isMalloc = false;
	if(thisBuffer->device_type() == base::DeviceType::kDeviceCUDA) {
		data_ptr = (T*)malloc(sizeof(T) * this->size());
		isMalloc = true;
		cudaMemcpy(data_ptr, thisBuffer->ptr(), sizeof(T) * this->size(), cudaMemcpyDeviceToHost);
	}else {
		data_ptr = reinterpret_cast<T*>(thisBuffer->ptr());
	}
	auto max_stack = std::vector<float>(5, 0.f);
	for(size_t i = 0; i < this->size(); i++) {
		for(size_t j = 0; j < max_stack.size(); j++) {
			if((float)(data_ptr[i]) > max_stack[j]) {
				max_stack[j] = (float)(data_ptr[i]);
				break;
			}
		}
	}
	for(size_t i = 0; i < max_stack.size(); i++) {
		printf("%f\n", max_stack[i]);
		fflush(stdout);
	}
	if (isMalloc){free(data_ptr);}
}

} // namespace tensor
#endif //TENSOR_HPP

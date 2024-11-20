//
// Created by CSWH on 2024/11/17.
//
#include "tensor.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>

namespace tensor{
// 累乘
template<typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init){
  if(begin >= end){
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}
// 不同数据类型占用的字节数
static size_t data_type_size(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::kDataTypeFp32: {
      return 4;
    }
    case base::DataType::kDataTypeInt8: {
      return 1;
    }
    case base::DataType::kDataTypeInt32: {
      return 4;
    }
    default: {
      LOG(FATAL) << "Unknown data type size for " << int(data_type);
      return 0;
    }
  }
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (buffer_) {
    if (buffer_->device_type() != buffer->device_type()) {
      LOG(ERROR) << "The device type of the new buffer is different from the original one.";
    }
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}


/*
1. 构造tensor时, 需要分配内存的情况下, 初始化buffer
2. 有些情况需要重新分配内存时, 比如重新分配更大的内存, need_realloc = true
 */
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool need_realloc) {
    size_t byte_size = this->byte_size();

    if(buffer_ && byte_size <= buffer_ -> byte_size() && !need_realloc) {
      return true;
    }
    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    return true;
}

/*
主要用于为外部数据创建buffer
在构造函数中, 到了这一步, 数据是一定有的, 要么需要管理, 要么不需要管理
需要管理会重新创建新buffer. 所以外部数据一般管理不了.
 */
void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                         base::DataType data_type,
                         bool need_alloc,
                         void* ptr) {
  if(!alloc && !need_alloc) { // 不需要管理
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_,
                                       nullptr,// 分配为空
                                       ptr,// 外部指针
                                       true); // 不需要管理
    this->buffer_ = buffer;
  } else { // 需要管理, 重分配.
    allocate(alloc, true);
  }
}


// 一维张量构造
Tensor::Tensor(base::DataType data_type,
               int32_t dim0,
               bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc,
               void* ptr)
              : data_type_(data_type){
  dims_.push_back(dim0);
  size_ = dim0;
  if(need_alloc && alloc){ // 需要分配, 需要管理
    allocate(alloc);
  } else { // 不需要分配 or 不需要管理
    if(ptr != nullptr){ // 外部数据, 一定不需要分配. 可能性只剩下 需要管理 or 不需要管理
      CHECK(need_alloc = false)
        << "The need_alloc is is true when ptr parameter is not a null pointer.";
      // 需要管理 or 不需要管理
      init_buffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}
// 二维张量的构造
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                 std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
      : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}
// 三维张量的构造
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
                 std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
      : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}
// 四维张量的构造
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                 bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
      : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

// 使用vector初始化向量形状的构造
Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
                 std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
      : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

/*
Tensor的设备类型由buffer决定. buffer的设备类型由allocator决定.
如果没有allocator, 要通过set_device_type手动确定.
 */
base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}
void Tensor::set_device_type(base::DeviceType device_type) const {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}


size_t Tensor::size() const { return this->size_; }
int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }
base::DataType Tensor::data_type() const { return data_type_; }
size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }
bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}
const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }
std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }


// 得到某维度上的大小
int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

/*
  修改维度影响的只是寻找数据的方法, 这些方法由dims_和size_决定.
 */
void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  if(!buffer_){
    this->dims_ = dims;
    this->size_ = size;
    return;
  }
  if (size > size_) {// 如果形状变大
    // 创建buffer的时候会自己申请内存
    auto new_buffer = std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_),
                                                     buffer_->allocator());
    new_buffer->copy_from(buffer_.get());
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}
// 某个维度上的一个步伐的stride，应该是其后维度大小的累乘
std::vector<size_t> Tensor::strides() const{
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (size_t i = 0; i < dims_.size(); ++i) {
      size_t stride = reduce_dimension(dims_.begin()+1+i, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}


void Tensor::to_cuda(){
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();
  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if(device_type == base::DeviceType::kDeviceCPU){
    size_t byte_size = this->byte_size();
    auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
    cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size,
                     base::MemcpyKind::kMemcpyCPU2CUDA);
    this->buffer_ = cu_buffer;
    // 设置Device Type
    this->buffer_->set_device_type(base::DeviceType::kDeviceCUDA);
  } else {
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}

void Tensor::to_cpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();

  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    size_t byte_size = this->byte_size();
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
    cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      base::MemcpyKind::kMemcpyCUDA2CPU);
    this->buffer_ = cpu_buffer;
    this->buffer_->set_device_type(base::DeviceType::kDeviceCPU);
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

/*
Tensor通过buffer持有数据, 清空buffer就清空了数据
 */
void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;
}

Tensor Tensor::clone() const {
  Tensor new_tensor = *this; // 解引用，拷贝构造
  size_t byte_size = this->byte_size();

  auto allocator = buffer_->allocator();
  new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
  new_tensor.buffer_->copy_from(buffer_.get());
  return new_tensor;
}

}// end namespace tensor
//
// Created by CSWH on 2024/10/28.
//

#ifndef ALLOC_HPP // if not defined如果没有被包括, 避免重复包含
#define ALLOC_HPP // define
#include <memory>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "base.hpp"

namespace base {


/*
 * 父类中最好有一个虚析构函数, 由于默认析构函数是非虚的, 通过子类new一个父类对象, 销毁时只调用父类的默认析构函数
 * 但是我们没有在子类中定义更多变量, 所以就无所谓了
 */
// 抽象类(接口)
class DeviceAllocator {
public:
    // 构造函数
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){};

    //查看设备类型
    virtual DeviceType device_type() const {return device_type_;}

    // 施放指针内存
    virtual void release(void* ptr) const = 0; // 纯虚函数, 子类必须有自己的实现

    // 申请内存
    virtual void* allocate(size_t byte_size) const = 0;

    // 数据移动
    virtual void memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind) const;

    // 设置初值
    virtual void memset_zero(void* ptr, size_t byte_size) const;

private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};


class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

// 单例模式
class CPUDeviceAllocatorFactory{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance(){
        if(instance == nullptr){
        instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
          instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

}//end namespace
#endif //ALLOC_HPP


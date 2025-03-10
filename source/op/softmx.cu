//
// Created by CSWH on 2025/3/10.
//
#include"softmax.cuh"
namespace op {
SoftmaxLayer::SoftmaxLayer(base::DeviceType device_type)
    : Layer(base::LayerType::kLayerSoftmax, "Softmax") {
    reset_input_size(1);
    reset_output_size(1);
}

bool SoftmaxLayer::forward() {
    auto input1 = this->get_input(0);
    auto output = this->get_output(0);
    kernel::softmax_kernel_cu(input1, output);
    return true;
}
}

namespace kernel {
__global__ void online_softmax_gpu(float *input, float *output, int length) {
    __shared__ float m;
    __shared__ float l;

    m = -FLT_MAX;
    l = 0;
    __syncthreads();

    int package_size = 4;
    int package_id = threadIdx.x + blockIdx.x * blockDim.x;
    int package_num = length / package_size;

    // online计算
    // 先在当前thread负责的package中online, 然后在共享内存中online
    float m_thread = -FLT_MAX;
    float l_thread = 0;
    for (int i = package_id; i < package_num; i += blockDim.x) {
        int package_start = i * package_size;
        float4& data = reinterpret_cast<float4 &>(input[package_start]);
        float temp = m_thread;

        m_thread = max(m_thread, data.x);
        m_thread = max(m_thread, data.y);
        m_thread = max(m_thread, data.z);
        m_thread = max(m_thread, data.w);

        l_thread *= expf(temp - m_thread);
        l_thread += expf(data.x - m_thread) + expf(data.y - m_thread) + expf(data.z - m_thread) + expf(data.w - m_thread);
    }
    // reduce m和l
    m = max(m, m_thread);
    __syncthreads();
    float add = l_thread * expf(m_thread - m);
    atomicAdd(&l, add); // 需要用原子操作或者block reduce
    __syncthreads();

    // 更新
    for (int i = package_id; i < length; i += blockDim.x) {
        int package_start = i * package_size;
        auto& data = reinterpret_cast<float4 &>(input[package_start]);
        auto& output_data = reinterpret_cast<float4&>(output[package_start]);

        output_data.x = expf(data.x - m) / l;
        output_data.y = expf(data.y - m) / l;
        output_data.z = expf(data.z - m) / l;
        output_data.w = expf(data.w - m) / l;
    }
}
void softmax_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& output) {
    int size = static_cast<int32_t>(input1.size());

    online_softmax_gpu<<<1, 128>>>(
            const_cast<float*>(input1.ptr<float>()), const_cast<float*>(output.ptr<float>()), size);
}
}
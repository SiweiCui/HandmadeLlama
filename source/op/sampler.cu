//
// Created by CSWH on 2024/11/19.
//
# include "sampler.cuh"
#include "alloc.hpp"
#include<random>

namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size) {
    size_t next = kernel::argmax_kernel_cu(logits, size);
    return next;
}

TopKSampler::TopKSampler(size_t k) : k_(k){
    // 初始化随机数引擎，使用默认的随机种子
    std::random_device rd;
    gen = std::mt19937(rd());
    // 初始化均匀分布
    dis = std::uniform_real_distribution<float>(0.f, 1.f);
}


size_t TopKSampler::sample(const float* logits, size_t size) {
    auto rand_num = dis(gen);
    size_t next = kernel::topk_kernel_cu(logits, size, k_, rand_num);
    return next;
}

}

namespace kernel {
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
    float tmp_val;
    size_t tmp_ptr;
    unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
    for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
        tmp_val = __shfl_down_sync(mask, val, k, warpSize);
        tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
        if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
        if (tmp_val > val) {
          val = tmp_val;
          ptr = tmp_ptr;
        } else if (tmp_val == val && tmp_ptr < ptr) {
          ptr = tmp_ptr;
        }
    }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                  size_t* shared_ptr) {
      int lane_id = threadIdx.x % warpSize;
      int warp_id = threadIdx.x / warpSize;

      warp_reduce_argmax(val, ptr);

      __syncthreads();
      if (lane_id == 0) {
      shared_value[warp_id] = val;
      shared_ptr[warp_id] = ptr;
      }

      __syncthreads();
      if (threadIdx.x < blockDim.x / warpSize) {
      val = shared_value[lane_id];
      ptr = shared_ptr[lane_id];
      } else {
      val = 0;
      ptr = SIZE_MAX;
      }

      if (warp_id == 0) {
      warp_reduce_argmax(val, ptr);
      }
}

__global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) {
    __shared__ size_t shared_max_ptr[32];
    __shared__ float shared_max_value[32];
    uint32_t tid = threadIdx.x;
    if (tid >= size) {
        return;
    }

    size_t max_index = threadIdx.x;
    float max_value = input_ptr[max_index];
    for (size_t i = tid; i < size; i += blockDim.x) {
        if (input_ptr[i] > max_value) {
          max_index = i;
          max_value = input_ptr[i];
        }
    }

    block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
    __syncthreads();
    if (threadIdx.x == 0) {
        *output_idx = max_index;
    }
}

size_t argmax_kernel_cu(const float* input_ptr, size_t size) {
    std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
    size_t output_index = 0;
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, index);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
    return output_index;
}

// 2050一个SM的共享内存最大是100KB, 能容纳25*2^10个浮点数, 远远大于词库. 放心使用共享内存
__global__ void topk_kernel_fp32(float* data, size_t size, size_t K, size_t* output_index, float rand_num) {
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* topk_prob_cache = shared_mem;
    auto topk_idx_cache = reinterpret_cast<size_t *>(shared_mem + K+K%2);

    // 动员所有线程并行处理长向量
    // 0号线程: 0, 512, 1024, ...
    // ...
    // 511号线程: 511, 1023, ...
    for(size_t i = tid; i < size; i += blockDim.x) {
        // 对于每个负责的元素, 遍历一遍数组, 计数比这个元素大的有多少
        float data_i = data[i];
        int count_greater = 0;
        for(size_t j = 0; j < size&&count_greater<K; ++j) {
            if(data[j] > data_i) {
                count_greater++;
            }
        }
        // 如果count_greater<K, 将其记录在topk cache的第 K-1-count_greater位置
        if(count_greater < K) {
            topk_idx_cache[K-1-count_greater] = i;
            topk_prob_cache[K-1-count_greater] = data_i;
        }
    }
    __syncthreads();

    // 动员所有线程做归一化
    __shared__ float sum_prob;
    sum_prob = 0.f;
    for(int i = tid; i < K; i+= blockDim.x) {
        atomicAdd(&sum_prob, topk_prob_cache[i]);
    }
    __syncthreads();
    for(int i = tid; i < K; i += blockDim.x) {
        topk_prob_cache[i] /= sum_prob;
    }
    __syncthreads();

    // 概率累加 & 选择
    if(threadIdx.x == 0) {
        /**
        float tmp_for_show = 0.f;
        for(int i = 0; i < K; ++i) {
            tmp_for_show += topk_prob_cache[i];
            printf("idx: %lu, real prob: %f, sum_prob: %f,  nmlzd prob: %f, acc prob: %f, rand_num: %f\n", topk_idx_cache[i], data[topk_idx_cache[i]], sum_prob, topk_prob_cache[i], tmp_for_show, rand_num);
        }
        **/
        float tmp = 0.f;
        for(int i = 0; i < K; ++i) {
            tmp += topk_prob_cache[i];
            if(tmp > rand_num) {*output_index = topk_idx_cache[i];break;}
            topk_prob_cache[i] = tmp;
        }
    }

}


size_t topk_kernel_cu(const float* input_ptr, size_t size, size_t K, float rand_num) {
    std::shared_ptr<base::DeviceAllocator> alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
    size_t output_index = 0;
    size_t align_K = K+K%2; // 内存对齐
    topk_kernel_fp32<<<1, 512, align_K*sizeof(float)+align_K*sizeof(size_t)>>>(const_cast<float*>(input_ptr), size, K, index, rand_num);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);

    return output_index;
}
}  // namespace kernel
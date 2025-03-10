#include<gtest/gtest.h>
#include<op/mha.cuh>
#include<tensor.hpp>


TEST(test_attention, test1) {
    int pos = 9; // 从0开始计数
    int layer_idx = 9; // 从0开始计数
    int layer_num = 22;
    int seq_len = 2048;
    int head_num = 32;
    const int head_size = 64;
    int kv_head_num = 4;
    int kv_mul = 8; // 一个query头对应多少kv头
    int kv_dim = 256; // KV Cache中seq之间的step

    // 所需数据
    float* q = (float*)malloc(sizeof(float) * head_num * head_size);
    float* k_cache = (float*)malloc(sizeof(float) * layer_num * seq_len * kv_dim);
    float* v_cache = (float*)malloc(sizeof(float) * layer_num * seq_len * kv_dim);
    for (int i = 0; i < head_num*head_size; i++) {
        q[i] = (rand()%10)/10.f;
    }
    // 补充0~9的k和v
    for (int i = 0; i < (pos+1)*kv_dim; i++) {
        k_cache[layer_idx*seq_len*kv_dim + i] = (rand()%10)/10.f;
        v_cache[layer_idx*seq_len*kv_dim + i] = (rand()%10)/10.f;
    }

    // 创建空间以及数据转移
    float* score_gpu, *output_gpu;
    float* q_gpu, *k_cache_gpu, *v_cache_gpu;
    cudaMalloc(&score_gpu, sizeof(float) * head_num * seq_len);
    cudaMalloc(&output_gpu, sizeof(float) * head_num * head_size);
    cudaMalloc(&q_gpu, sizeof(float) * head_num * head_size);
    cudaMalloc(&k_cache_gpu, sizeof(float) * layer_num * seq_len * kv_dim);
    cudaMalloc(&v_cache_gpu, sizeof(float) * layer_num * seq_len * kv_dim);

    cudaMemcpy(q_gpu, q, sizeof(float) * head_num * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_cache_gpu, k_cache, sizeof(float) * layer_num * seq_len * kv_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(v_cache_gpu, v_cache, sizeof(float) * layer_num * seq_len * kv_dim, cudaMemcpyHostToDevice);


    // 执行核函数
    kernel::multi_head_attention_kernel<<<head_num, 128>>>(pos, seq_len, q_gpu, score_gpu, output_gpu,
        k_cache_gpu, v_cache_gpu, kv_dim, kv_mul, head_num, head_size, layer_idx * seq_len * kv_dim);

    // 展示结果
    tensor::Tensor q_tensor(base::DataType::kDataTypeFp32, head_num*head_size, false, nullptr, q_gpu);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_num*head_size, false, nullptr, output_gpu);
    q_tensor.set_device_type(base::DeviceType::kDeviceCUDA);
    output_tensor.set_device_type(base::DeviceType::kDeviceCUDA);
    printf("q:\n");
    q_tensor.show_digits<float>(100);
    printf("\n");
    printf("mha output:\n");
    output_tensor.show_digits<float>(100);
    printf("\n");


    // flash attention
    float* output_flash;
    cudaMalloc(&output_flash, sizeof(float) * head_num * head_size);
    kernel::flash_attention_kernel<<<head_num, 128, (head_size*4+2)*sizeof(float)>>>(pos, seq_len, q_gpu, score_gpu, output_flash,
        k_cache_gpu, v_cache_gpu, kv_dim, kv_mul, head_num, head_size, layer_idx * seq_len * kv_dim);

    tensor::Tensor output_flash_tensor(base::DataType::kDataTypeFp32, head_num*head_size, false, nullptr, output_flash);
    output_flash_tensor.set_device_type(base::DeviceType::kDeviceCUDA);
    printf("flash attn output:\n");
    output_flash_tensor.show_digits<float>(100);

    output_tensor.to_cpu();
    output_flash_tensor.to_cpu();
    for(int i = 0; i < head_num*head_size; i++) {
        if(output_tensor.index<float>(i) != output_flash_tensor.index<float>(i)) {
            printf("output_gpu[%d] = %f, output_flash[%d] = %f \n", i, output_tensor.index<float>(i), i, output_flash_tensor.index<float>(i));
        }
    }

}
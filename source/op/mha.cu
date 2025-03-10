//
// Created by CSWH on 2024/11/19.
//

#include "mha.cuh"
#include "tensor.hpp"
#include <cub/cub.cuh>

namespace op {
MultiHeadAttention::MultiHeadAttention(int32_t layer_index,
									   int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
									   int32_t head_num, int32_t head_size)
	: Layer(base::LayerType::kLayerMHA, "MultiHead"),
	  layer_index_(layer_index),
	  kv_mul_(kv_mul), // kv头需要"复制"的倍数
	  kv_dim_(kv_dim),
	  seq_len_(seq_len),
	  head_num_(head_num),
	  head_size_(head_size), attention_config_(base::AttentionConfig::kFlashAttention){
	reset_input_size(5);
	reset_output_size(1);
}

MultiHeadAttention::MultiHeadAttention(int32_t layer_index,
									   int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
									   int32_t head_num, int32_t head_size, base::AttentionConfig attention_config)
		: Layer(base::LayerType::kLayerMHA, "MultiHead"),
		  layer_index_(layer_index),
		  kv_mul_(kv_mul), // kv头需要"复制"的倍数
		  kv_dim_(kv_dim),
		  seq_len_(seq_len),
		  head_num_(head_num),
		  head_size_(head_size), attention_config_(attention_config){
	reset_input_size(5);
	reset_output_size(1);
}


bool MultiHeadAttention::forward() {
	const tensor::Tensor& mha_out = this->get_output(0);
	const tensor::Tensor& query_tensor = this->get_input(0);
	const tensor::Tensor& score_tensor = this->get_input(1);
	const tensor::Tensor& key_cache_tensor = this->get_input(2);
	const tensor::Tensor& value_cache_tensor = this->get_input(3);

	kernel::mha_kernel_cu(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
										 head_size_, mha_out, query_tensor, score_tensor,
										 key_cache_tensor, value_cache_tensor);
	return true;
}

void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }

}

namespace kernel {

constexpr static int thread_num = 256;

__device__ void softmax_gpu(float* __restrict__ x, int size) {
	 int tid = threadIdx.x;
	 int step = blockDim.x;

	 // find max value (for numerical stability)
	 // this should be FLT_MAX, not 0 !!!!
	 // otherwise, the softmax may be occur nan when head_dim < 128 threads
	 float max_val = tid < size ? x[tid] : -FLT_MAX;
	 for (int i = tid + step; i < size; i += step) {
	   if (x[i] > max_val) {
	     max_val = x[i];
	   }
	 }
	 using BlockReduce = cub::BlockReduce<float, thread_num>;
	 __shared__ BlockReduce::TempStorage temp;
	 __shared__ float shared_val;
	 max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
	 if (threadIdx.x == 0) {
	   shared_val = max_val;
	 }
	 __syncthreads();
	 max_val = shared_val;

	 float sum = 0.0f;
	 for (int i = tid; i < size; i += step) {
	   x[i] = expf(x[i] - max_val);
	   sum += x[i];
	 }
	 sum = BlockReduce(temp).Sum(sum);
	 if (threadIdx.x == 0) {
	   shared_val = sum;
	 }
	 __syncthreads();
	 sum = shared_val;

	 for (int i = tid; i < size; i += step) {
	   x[i] /= sum;
	 }

	// if(tid == 0) {
	// 	printf("mha, head_id: %d, l = %f, m = %f\n", blockIdx.x, sum, max_val);
	// }

}

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
	 int head = blockIdx.x;
	 if (head >= head_num) {
	   return;
	 }

	float scale = 1.f / sqrtf(head_size);
	// head, head_dim
	float* query_head = query + head * head_size;
	// head, head_dim
	float* score_head = score_ptr + head * seq_len;
	int head_offset = (head / kv_mul) * head_size; // 在KV中的偏置

	// 本线程会处理一个head的许多seq, 每个seq相隔blockDim.x, 这是在项目中处理向量时常见的
	for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
		// (layer, seq, head, head_dim), t*kv_dim是seq的偏置.
		// 这行代码定位到本次要处理的seq的key, 即k_t
		float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

		float score = 0.0f;

		// score[t] = <q, k_t> * scale
		#pragma unroll
		for (int i = 0; i < head_size; i += 4) {
		  float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
		  float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
		  score += key_head_float4.x * query_head_float4.x;
		  score += key_head_float4.y * query_head_float4.y;
		  score += key_head_float4.z * query_head_float4.z;
		  score += key_head_float4.w * query_head_float4.w;
		}
		score *= scale;
		score_head[t] = score;

	}
	 __syncthreads();

	 softmax_gpu(score_head, pos + 1);// 因为有mask, 计算下三角的softmax即可.
	 __syncthreads();

	// 我们得到了score = (score[0], ..., score[pos])
	// 假设有v0, v1, v2, ..., v_pos(都是行向量), output = score[0]*v0 + ... + score[pos]*v_pos.
	// 下面的是逐个元素计算的方法.
	float* output_head = output + head * head_size;
	// output阶段, 每个线程处理一个output的一个元素.
	for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
		float value = 0.0f;
		#pragma unroll
		// 每个元素是一个加权
		for (int t = 0; t <= pos; t++) {
			float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
			float score = score_head[t];
			value += score * value_head[i];
		}
		output_head[i] = value;
	}
	// if (threadIdx.x == 0) {
	// 	printf("mha, head_id = %d, kv_head_offset = %d, q_head_offset = %d\n", blockIdx.x, head_offset, head*head_size);
	// }
}

//一个block处理一个head
__global__ void flash_attention_kernel(int32_t pos, int32_t seq_len, float* query,
												float* score_ptr, float* output, float* key_cache,
												float* value_cache, int32_t kv_dim, int32_t kv_mul,
												int32_t head_num, int32_t head_size,
												int32_t layer_offset) {
	int head_id = blockIdx.x;
	if (head_id >= head_num) {return;}
	int kv_head_offset = (head_id / kv_mul) * head_size;
	int q_head_offset = head_id * head_size;
	int tid = threadIdx.x;

	float scale = 1.f / sqrtf(head_size);

	extern __shared__ float shared_mem[];
	float* shared_q = shared_mem;
	float* shared_k_t = shared_q + head_size;
	float* shared_v_t = shared_k_t + head_size;
	float* shared_o_t = shared_v_t + head_size;
	float* m = shared_o_t + head_size;
	float* l = m+1;

	// 初始化共享变量
	*m = -FLT_MIN;
	*l = 0;
	for(int h = 0; h < head_size; h++) {
		shared_o_t[h] = 0.0f;
	}
	// __syncthreads();

	// 并行load query, output
	float* q_t = query + q_head_offset;
	for(int i = tid; i < head_size; i += blockDim.x) {
		shared_q[i] = q_t[i];
	}
	__syncthreads();


	for (int t = 0; t <= pos; t ++) { // 此处不做并行化
		// 并行load k_t, v_t
		float* k_t = key_cache + layer_offset + t * kv_dim + kv_head_offset;
		float* v_t = value_cache + layer_offset + t * kv_dim + kv_head_offset;
		for(int i = tid; i < head_size; i += blockDim.x) {
			shared_k_t[i] = k_t[i];
			shared_v_t[i] = v_t[i];
		}
		__syncthreads();

		// 内积<q, k_t>
		float s_t = 0.f;
		for (int j = 0; j < head_size; j++) {
			s_t += shared_q[j] * shared_k_t[j];
		}
		s_t *= scale; // block中每个线程都有一份, 而且一样大

		// 更新m和l
		float m_old = *m;
		if(tid == 0) { // 赋值, 让多个线程做也没关系, 这里让一个线程去做
			*m = max(m_old, s_t);
			*l = expf(m_old - *m) * *l + expf(s_t - *m);
		}
		__syncthreads();

		// 动员所有线程更新O
		float p_t = expf(s_t - *m);
		for(int i = tid; i < head_size; i += blockDim.x) {
			shared_o_t[i] = expf(m_old - *m) * shared_o_t[i] + p_t * shared_v_t[i]; // 线程之间不会有写冲突
		}
		__syncthreads();
	}

	// 用最终的l更新
	float* o = output + q_head_offset;
	for(int i = tid; i < head_size; i += blockDim.x) {
		o[i] = shared_o_t[i] / *l;
	}

}


void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, const int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor) {
	int32_t layer_offset = layer_index * seq_len * kv_dim;
	float* query = const_cast<float*>(query_tensor.ptr<float>());
	float* score = const_cast<float*>(score_tensor.ptr<float>());
	float* output = const_cast<float*>(mha_out.ptr<float>());

	float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
	float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

	flash_attention_kernel<<<head_num, thread_num, (head_size*4+2)*sizeof(float)>>>(
	  pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
	  head_size, layer_offset);
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
					   int32_t kv_dim, int32_t kv_mul, const int32_t head_size, const tensor::Tensor& mha_out,
					   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
					   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
					   base::AttentionConfig attention_config) {
	int32_t layer_offset = layer_index * seq_len * kv_dim;
	float* query = const_cast<float*>(query_tensor.ptr<float>());
	float* score = const_cast<float*>(score_tensor.ptr<float>());
	float* output = const_cast<float*>(mha_out.ptr<float>());

	float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
	float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

	if(attention_config == base::AttentionConfig::kGQA) {
		multi_head_attention_kernel<<<head_num, thread_num>>>( // 一个block处理一个head
		  pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
		  head_size, layer_offset);
	}else {
		flash_attention_kernel<<<head_num, thread_num, (head_size*4+2)*sizeof(float)>>>(
		  pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
		  head_size, layer_offset);
	}
}

}  // namespace kernel
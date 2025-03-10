//
// Created by CSWH on 2024/11/17.
//

#ifndef BASE_HPP
#define BASE_HPP
#include <memory>

namespace base {
enum class MemcpyKind{
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3
};

enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    }
    if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    }
    if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    }
    return 0;
}

enum class ModelType: uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};
enum class ModelBufferType {
	kInputTokens = 0,
	kInputEmbeddings = 1,
	kOutputRMSNorm = 2,
	kKeyCache = 3,
	kValueCache = 4,
	kQuery = 5,
	kInputPos = 6,
	kScoreStorage = 7,
	kOutputMHA = 8,
	kAttnOutput = 9,
	kW1Output = 10,
	kW2Output = 11,
	kW3Output = 12,
	kFFNRMSNorm = 13,
	kForwardOutput = 15,
	kForwardOutputSoftmax = 16,

	kSinCache = 17,
	kCosCache = 18,
};

enum class TokenizerType {
	kEncodeUnknown = -1,
	kEncodeSpe = 0,
	kEncodeBpe = 1,
};

struct ModelConfig {
	int32_t dim = 0;
	int32_t hidden_dim = 0;
	int32_t layer_num = 0;
	int32_t head_num = 0;
	int32_t kv_head_num = 0;
	int32_t vocab_size = 0;
	int32_t seq_len = 0;
};

struct TransformerConfig {
	int32_t kv_dim_ = 0;
	int32_t kv_mul_ = 0;
	int32_t head_size_ = 0;
	int32_t vocab_size_ = 0;

	int32_t dim_ = 0;
	int32_t hidden_dim_ = 0;
	int32_t layer_num_ = 0;
	int32_t head_num_ = 0;
	int32_t kv_head_num_ = 0;
	int32_t seq_len_ = 0;
	bool is_shared_weight_ = false;
};

enum class LayerType : uint8_t {
	kLayerUnknown = 0,
	kLayerLinear = 1,
	kLayerEncode = 2,
	kLayerEmbedding = 3,
	kLayerRMSNorm = 4,
	kLayerMatmul = 5,
	kLayerRoPe = 6,
	kLayerMHA = 7,
	kLayerSoftmax = 8,
	kLayerAdd = 9,
	kLayerSwiGLU = 10,
};

enum class AttentionConfig : uint8_t {
	kGQA = 0,
	kFlashAttention = 1
};

enum class SamplerConfig : uint8_t {
	kArgMaxSampler = 0,
	kTopkSampler = 1
};

}


#endif //BASE_HPP

//
// Created by CSWH on 2024/11/18.
//

#ifndef ROW_MODEL_DATA_HPP
#define ROW_MODEL_DATA_HPP

# include <cstddef>
# include <cstdint>

namespace model {
	// 持有mmap的权重内存指针.
struct RawModelData {
	~RawModelData();
	int32_t fd = -1;
	size_t file_size = 0;
	void* data = nullptr;
	void* weight_data = nullptr;

	virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32:RawModelData {
	const void* weight(size_t offset) const override;
};

struct RawModelDataInt8:RawModelData {
	const void* weight(size_t offset) const override;
};
}

#endif //ROW_MODEL_DATA_HPP

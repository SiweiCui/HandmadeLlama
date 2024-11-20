//
// Created by CSWH on 2024/11/18.
//
#include "row_model_data.hpp"
#include <sys/mman.h>
#include <unistd.h>
namespace model {
	RawModelData::~RawModelData() {
		if (data != nullptr && data != MAP_FAILED) {
			// 释放mmap的内存区域
			munmap(data, file_size);
			data = nullptr;
		}
		if (fd != -1) {
			close(fd);
			fd = -1;
		}
	}

	const void *RawModelDataFp32::weight(size_t offset) const {
		return static_cast<const float*>(data + offset);
	}

	const void* RawModelDataInt8::weight(size_t offset) const {
		return static_cast<const int8_t*>(data + offset);
	}

}
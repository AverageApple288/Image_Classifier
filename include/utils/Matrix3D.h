//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_MATRIX_H
#define IMAGE_CLASSIFIER_MATRIX_H

#include <vector>
#include <iostream>

class Matrix3D {
private:
	size_t depth_{};
	size_t rows_{};
	size_t cols_{};
	std::vector<float> data_;

public:
	// Constructor
	Matrix3D(size_t depth, size_t rows, size_t cols, float initial_value = 0.0f);
	Matrix3D();

	// Getters
	[[nodiscard]] size_t depth() const;
	[[nodiscard]] size_t rows() const;
	[[nodiscard]] size_t cols() const;

	// Index operators
	float& operator()(size_t d, size_t r, size_t c) noexcept {
		return data_[(d * rows_ + r) * cols_ + c];
	}

	const float& operator()(size_t d, size_t r, size_t c) const noexcept {
		return data_[(d * rows_ + r) * cols_ + c];
	}

	// Checked version if needed for explicit validation
	float& at(size_t d, size_t r, size_t c) {
		if (d >= depth_ || r >= rows_ || c >= cols_) {
			throw std::out_of_range("Matrix3D index out of bounds");
		}
		return data_[(d * rows_ + r) * cols_ + c];
	}

	Matrix3D apply_relu();

	[[nodiscard]] const std::vector<float>& get_flat_data() const {
		return data_;
	}

	// Utility
	void print() const;
};



#endif //IMAGE_CLASSIFIER_MATRIX_H

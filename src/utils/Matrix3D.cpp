//
// Created by fabian on 16/08/2026.
//

#include "../../include/utils/Matrix3D.h"

#include <stdexcept>
#include <iostream>

Matrix3D::Matrix3D(const size_t depth, const size_t rows, const size_t cols, const float initial_value)
	: depth_(depth), rows_(rows), cols_(cols),
	  data_(depth * rows * cols, initial_value) {}

Matrix3D::Matrix3D() = default;

size_t Matrix3D::depth() const { return depth_; }
size_t Matrix3D::rows() const { return rows_; }
size_t Matrix3D::cols() const { return cols_; }

Matrix3D Matrix3D::apply_relu() {
	auto activated_matrix = Matrix3D(depth_, rows_, cols_, 0.0f);

	for (size_t d = 0; d < depth_; ++d) {
		for (size_t r = 0; r < rows_; ++r) {
			for (size_t c = 0; c < cols_; ++c) {
				if ((*this)(d, r, c) < 0.0f) {
					activated_matrix(d, r, c) = 0.0f;
				}
			}
		}
	}

	return activated_matrix;
}

void Matrix3D::print() const {
	for (size_t d = 0; d < depth_; ++d) {
		std::cout << "Depth " << d << ":\n";
		for (size_t r = 0; r < rows_; ++r) {
			for (size_t c = 0; c < cols_; ++c) {
				std::cout << (*this)(d, r, c) << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}

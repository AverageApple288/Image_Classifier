//
// Created by fabian on 16/08/2026.
//

#include "../../include/utils/Matrix3D.h"

#include <stdexcept>
#include <iostream>

Matrix3D::Matrix3D(const size_t depth, const size_t rows, const size_t cols, const double initial_value)
	: depth_(depth), rows_(rows), cols_(cols),
	  data_(depth * rows * cols, initial_value) {}

size_t Matrix3D::depth() const { return depth_; }
size_t Matrix3D::rows() const { return rows_; }
size_t Matrix3D::cols() const { return cols_; }

double& Matrix3D::operator()(const size_t d, const size_t r, const size_t c) {
	if (d >= depth_ || r >= rows_ || c >= cols_) {
		throw std::out_of_range("Matrix3D index out of bounds");
	}
	return data_[d * (rows_ * cols_) + r * cols_ + c];
}

const double& Matrix3D::operator()(const size_t d, const size_t r, const size_t c) const {
	if (d >= depth_ || r >= rows_ || c >= cols_) {
		throw std::out_of_range("Matrix3D index out of bounds");
	}
	return data_[d * (rows_ * cols_) + r * cols_ + c];
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
//
// Created by fabian on 13/01/2026.
//

#include "../include/Matrix.h"

#include <stdexcept>

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
	data.resize(r * c, 0.0f);
}

// 2D to 1D Mapping
// Index = (Row * Width) + Col
float& Matrix::at(int r, int c) {
	if (r < 0 || r >= rows || c < 0 || c >= cols) {
		throw std::out_of_range("Matrix index out of bounds");
	}
	return data[r * cols + c];
}

const float& Matrix::at(int r, int c) const {
	if (r < 0 || r >= rows || c < 0 || c >= cols) {
		throw std::out_of_range("Matrix index out of bounds");
	}
	return data[r * cols + c];
}

Matrix Matrix::matmul(const Matrix& A, const Matrix& B) {
	if (A.cols != B.rows) {
		throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
	}

	Matrix C(A.rows, B.cols);

	// Standard O(N^3) Matrix Multiplication
	// Optimization Tip: Swapping loops to i-k-j is faster due to CPU cache locality
	for (int i = 0; i < A.rows; ++i) {
		for (int k = 0; k < A.cols; ++k) {
			float a_val = A.at(i, k); // Cache this value
			for (int j = 0; j < B.cols; ++j) {
				// C[i][j] += A[i][k] * B[k][j]
				C.at(i, j) += a_val * B.at(k, j);
			}
		}
	}

	return C;
}